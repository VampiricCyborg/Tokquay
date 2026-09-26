# Design notes

Decisions made while building Tokquay, with the reasoning. The brief's three
write-ups (block size, recompute vs. swap, chunked prefill) are at the end, each
with the numbers from the Phase 6 benchmark. All numbers quoted here are in
`results.md`, generated from the saved benchmark results.

## Phase 4: scheduling policy

### Prefill and decode: alternating steps, prefill first (not mixed)

Every step is either a **prefill-only** step (the newly admitted requests) or a
**decode-only** step (all running sequences). If any waiting request can be
admitted, the step is a prefill step; otherwise it is a decode step.

*Why not mix them in one step?* The model supports it (each sequence carries its
own start position and length, and a test covers a mixed batch). But a batch is a
rectangle: every row is padded to the longest row. A 200-token prompt next to 30
running sequences would make each of the 30 one-token decodes compute 200 padded
positions, roughly a 200x waste on those rows. Getting the benefit of mixing
without that waste needs ragged (flattened, variable-length) attention, which is
out of scope. Alternating keeps a decode step at exactly one token per sequence.

*What it costs:* running sequences stall while a prefill step runs, so
inter-token latency spikes whenever new requests arrive, and a steady stream of
arrivals can delay decoding until the batch or the pool is full. This is the
problem chunked prefill solves (see below); we accept it and the benchmark
should show it in the p99 inter-token latency.

### Admission: FCFS, no skipping

Admission walks the waiting queue in arrival order and stops at the first
request that does not fit (sequence limit, per-step token budget, or free
blocks). It never jumps a small request ahead of a large one that is waiting,
so a large request cannot be starved.

The per-step token budget (`max_num_batched_tokens`) is counted in **padded**
tokens, `batch size x longest prompt`, because that is what the model computes
and what bounds activation memory.

### Preemption: recompute, newest first, back to the front of the queue

When a decode step needs a block and none is free, the **newest** running
sequence is evicted: its blocks are freed and it re-enters the waiting queue at
the **front**. It keeps the tokens it has generated; on re-admission the prompt
plus all generated tokens are prefilled again ("recompute"), and generation
resumes.

* *Newest first* protects the work already invested in older requests and keeps
  the oldest request always making progress.
* *Front of the queue* keeps the waiting queue in arrival order (a test checks
  this as an invariant under random workloads).
* *Recompute instead of swapping to CPU:* swapping needs a second (host) pool,
  block copies over PCIe in both directions, and bookkeeping for sequences that
  are "running but not resident". Recompute needs no extra machinery: the same
  prefill path that serves new requests serves preempted ones. The expectation
  was that at GPT-2 124M scale re-prefilling a few hundred tokens is cheap
  enough that this is the right trade. The Phase 6 benchmark confirms it: see
  "Recompute vs. swap" below.

### Rejecting requests up front

A request is refused at `add_request` if its worst case cannot be served:

* `prompt + max_tokens - 1` exceeds the model's context window (1024 positions);
* it needs more KV blocks than the whole pool holds;
* it needs more tokens than one step's budget, which would make a preempted
  copy of it impossible to re-prefill.

(`- 1` because the last sampled token is returned to the caller and never needs
K/V.) This is what rules out deadlock: every admitted request fits in an empty
pool, and the *oldest* running sequence is only ever evicted when it is the only
one left, at which point the whole pool is free for it. So the oldest request
always finishes, then the next, and so on.

### Watermark

While other sequences are running, admission keeps `watermark_blocks` (default 1)
blocks free. Without it a preempted sequence can be re-admitted into the last
free blocks, hit its neighbours' next block boundary, and be evicted again:
recomputing the same prompt over and over. The reserve does not apply to a
request that would run alone, so a request needing the whole pool is not
blocked by it.

## Phase 5: server and streaming

### The engine runs on its own thread, and only that thread touches it

`Engine.step()` is a blocking forward pass (tens of milliseconds, far more with a
big batch or a long prefill). Run as an asyncio task it would freeze every open
SSE stream and every new connection for the length of the step. So the loop runs
on one dedicated thread (PyTorch releases the GIL inside its ops, so the event
loop stays responsive while a step runs).

The alternatives were `run_in_executor(engine.step)` from an asyncio task, or a
lock around the engine. Both leave the scheduler's queues reachable from two
threads. Instead the engine thread is the *only* one that ever touches the engine,
scheduler or allocator. The event loop sends it commands through an inbox queue
("add this request", "abort that one") and gets tokens back through one
`asyncio.Queue` per request, filled with a single `call_soon_threadsafe` per step
(not per token). No locks, and the invariants the scheduler tests check hold
because there is exactly one writer.

Rejection checks (`rejection_reason`) depend only on constants (pool size, limits),
so the event loop runs them directly and answers 400 immediately instead of
round-tripping through the engine thread.

### Cancellation: a request nobody is waiting for must stop

If a client disconnects and the engine keeps decoding, the GPU works for nobody
and the request's KV blocks stay allocated, which under load starves real
requests. `Scheduler.abort` / `Engine.abort_request` remove a sequence from
wherever it is (waiting, preempted and waiting, or running) and return its blocks.

The streaming endpoint aborts in a `finally` around the token loop, which runs
whether the stream ends, the client leaves, or the task is cancelled. A plain JSON
request is not cancelled by the server when the client leaves, so that path checks
`request.is_disconnected()` once per token. Tests disable the abort and confirm both
disconnect tests fail, and check that after a disconnect the pool is whole again.

### A dead engine fails requests instead of hanging them

If `step()` raises, the engine state is inconsistent and there is no safe way to
continue. The thread records the failure, delivers `EngineError` to every in-flight
request and every command still in the inbox, and marks itself closed. Streams end
with an `event: error`, plain requests get 503, new requests get 503, and `/health`
returns 503. The alternative, a thread that dies quietly, leaves every client
waiting forever.

### Wire format

* `stream: false`: one JSON object (`text`, `token_ids`, `finish_reason`, token
  counts). `stream: true`: SSE, one `data:` event per token, the last one carries
  `finish_reason`. FastAPI's `EventSourceResponse` only supports endpoints that are
  generators, and the brief wants one endpoint with a `stream` flag, so the
  `StreamingResponse` is built by hand.
* Every token is an event, even when it produces no text (part of a multi-byte
  character), so clients and the Phase 6 benchmark can count tokens and timestamp
  each one.
* The EOS token is generated and counted (`token_ids` includes it, `finish_reason`
  is `stop`) but produces no text.
* Optional `seed` (reproducible sampling) and `ignore_eos` (fixed-length runs, which
  the benchmark needs) go beyond the brief's field list.

### Detokenizing a stream

GPT-2's byte-level BPE can spread one character over several tokens (a rocket emoji
is three). Decoding each token alone would emit U+FFFD for the incomplete pieces.
`IncrementalDetokenizer` decodes a small window before and after each new token and
emits only the added text, holding back while the window ends in U+FFFD; `flush`
emits the remainder at the end of the stream. Byte-level decoding is context-free,
so the concatenated deltas equal decoding everything at once (a test asserts it).

### Known limits

* A request that sits in the waiting queue produces no bytes until it is admitted,
  so a proxy with a short idle timeout could cut it. No keep-alive comments are sent.
* One process, one engine: uvicorn workers would each load the model and own a pool.
* A quick check with 20 concurrent requests (same workload, engine alone vs. through
  HTTP) found the server adding about 5% to the run time. That is one measurement
  on one machine, not a benchmark; Phase 6 should measure it properly.

## Phase 6: benchmark

### What is compared, and how it is kept fair

* **Equal KV memory.** Both systems get the same budget, 16,384 token slots
  (1.125 GiB at 72 KiB per slot for GPT-2 small in fp32). Tokquay turns it into
  1,024 blocks of 16; the baseline into 32 rows of `max_len` = 512 slots. That is
  the paper's claim made testable: with the same memory, how much can each serve?
  `max_len` = 512 is the workload's own upper bound (256 + 256), which favours the
  baseline: a server that does not know output lengths would reserve the model's
  1,024 positions per row and fit 16 rows, not 32.
* **The baseline is a real static batcher, built to be fair.** It takes up to N
  waiting requests when idle (work-conserving, FCFS), prefills them together, and
  decodes until the *slowest* finishes. Rows that finish early keep their slot and
  are fed a dummy token, as a batched `generate()` does. Nobody joins mid-batch.
  It reads each row's K/V with a slice, not a block-table gather, so it is not
  charged for a mechanism it does not use; both go through the same model code
  (`BatchedKV`), and both are checked token for token against the single-request
  reference in the tests.
* **Open loop.** Requests arrive when the Poisson process says, whether or not the
  system has kept up, and latency is measured from the *scheduled* arrival. A
  request that waits because the driver was busy inside a step is charged for it.
  The same seed gives the same prompts and lengths at every rate (only the gaps are
  rescaled), so runs at different rates are paired.
* **Timestamps.** A token is stamped when the step that produced it returns (each
  step ends in a GPU sync). ITL is the gap between consecutive tokens of one
  request, pooled over requests, so a preemption shows up as one long gap.
* **The metric code is tested against a hand calculation.** A scripted server on a
  virtual clock has TTFT, ITL, throughput, KV figures and stall classification worked
  out by hand in `tests/test_phase6_bench.py`. The work accounting is exact: the
  model's computed positions equal the necessary work plus exactly the positions
  preemption discarded.
* **KV memory is snapshotted at each step's peak**, after the forward pass and
  before finished sequences free their blocks. An earlier version snapshotted after
  the step and would have under-counted the peak.

### Two environment effects that distorted the first runs

Both were found because a number could not be right, and both were large enough to
change conclusions. They are the reason the benchmark takes the flags it does.

1. **GPU memory spill.** The first sweep showed Tokquay at 155 tok/s (2 requests/s)
   and 90 tok/s (3 requests/s), with a median TTFT of 80 ms but a p99 of 97 s at 2
   requests/s. That was worse than the baseline and far below its own saturated
   throughput. The tell was mean decode-step time (108 ms) *above* the p99 (93 ms):
   a few steps were extremely long. An instrumented replay showed those steps all happened while PyTorch's
   caching allocator had reserved 4,070 MiB, the whole 4 GiB card, with driver free
   memory at 0, although only 1,636 MiB was allocated. The paged gather makes large
   temporaries whose sizes change every step; the allocator fragments and grows,
   and on Windows the driver then spills to system RAM instead of failing. Steps
   took 0.6 to 1.5 s. Capping the allocator at 80% of VRAM
   (`torch.cuda.set_per_process_memory_fraction`) made it recycle memory: the same
   replay went from 478 to 687 tok/s and the p99 step from 674 to 126 ms. The cap is
   applied to both systems, and each run records its peak reserved memory and
   allocator retries (a table in `results.md`); the harness warns above 95% of VRAM.
   The replay was a one-off diagnostic, not part of the benchmark suite.
2. **CPU core placement.** Identical runs disagreed by up to 4x: the baseline's
   median ITL was 13.7 ms in one run and 44 to 54 ms in others, for the same full
   32-row batches. Decode steps at this size are bound by how quickly Python can
   launch kernels, so they run at the speed of the core they land on, and the
   i5-12450H has 4 performance and 4 efficiency cores. Pinning a fixed job to each
   set: decode step p50 was 26 ms on performance cores at high priority (31 ms at normal
   priority), 58 ms on efficiency cores, and 49 to 62 ms unpinned. The benchmark pins itself to the performance cores at
   high priority (`--cpu-affinity FF --high-priority`) and records both. Pinned,
   repeat runs agree within 1% at 4 requests/s, and in the calibration run the ITL p99
   fell from about 160 ms to about 50 ms. The results from before this fix were discarded.

What is left is at most a few percent of run-to-run noise: the 4 requests/s
configuration repeated to 571 vs 567 tok/s (Tokquay) and 545 vs 542 (baseline);
Tokquay's identical 8 requests/s configuration measured 804, 783 and 794 tok/s across
three experiments, the baseline's 754 and 702. Differences smaller than that are not
findings.

### What the benchmark shows, and what it does not

Below saturation the systems deliver the same throughput (the load is the limit) and
differ in latency: Tokquay's TTFT is 36 to 45 ms at the median against 2.6 to 4.8 s,
because a static batch admits nobody until it drains. With the full KV budget its
throughput advantage at saturation is small (5 to 9%, near the noise; 29% when all
requests arrive at once). The throughput advantage is clear when memory is scarce:
with a quarter of the budget (288 MiB) Tokquay serves 634 tok/s against the
baseline's 226, and is within 10% of the baseline given four times the memory.

Tokquay pays in inter-token latency. Its median ITL is 24 to 27 ms against 18 to
22 ms at 2 to 4 requests/s, and 56 to 66 ms against about 18 to 20 ms at 6 to 8. At
high load it runs 64 sequences per decode step to the baseline's 32, and prefill steps
stall decoding (below). That paged attention here is plain PyTorch, gathering K/V into
a copy each step where a fused kernel would not, is a plausible third cause that this
benchmark did not isolate; the brief rules out custom kernels.

## Recompute vs. swap

Recompute stays. At the tighter KV budgets (8,192 down to 2,048 slots, 8 requests/s)
Tokquay preempted 218 to 235 times per run, and 28 to 30% of all token positions the
model ran were recomputation (with the full budget: 12 preemptions, 1.8%). But
*all* prefill steps together, ordinary ones included, took only 7 to 15% of run time.
That is an upper bound on what recomputation cost, and so on what swapping could save
if the copies over PCIe were free, which they are not. The reason is that a step at
this size is bound by launching kernels, not by arithmetic: a prefill of about 140
padded tokens takes 25 ms, close to one decode step.

The throughput lost as the budget shrinks (794, 724, 634, 358 tok/s) tracks the drop
in concurrency (64, 41, 25, 14 sequences) more than the recomputation, which is what
any preemption scheme would pay, swapping included. This measures recompute only:
swap is not implemented, so nothing here shows it would be worse, only that there is
little for it to save at this scale.

## Chunked prefill (not implemented)

A long prompt is prefilled in one step, and the whole running batch waits for it.
Chunked prefill splits a long prompt into pieces of at most N tokens and spreads them
over several steps, mixed with decodes, so no step is long and decode latency stays
flat. It needs ragged attention (or per-row padding tricks) to avoid the padding
waste described above.

What the stalls cost here (Tokquay, prompts of 16 to 256 tokens): 5%, 11%, 24% and
28% of token gaps at 2, 4, 6 and 8 requests/s contained a prefill step, and those
gaps were 1.5 to 2 times as long as the others (median 49 to 91 ms against 23 to 62).
The longest prefill step observed was 81 ms, a few decode steps. A prefill of
about 1,870 padded tokens (all requests arriving at once, so the token budget binds)
took 117 ms, against 25 ms for about 140.

So for this workload the stalls are modest: chunking could at best bring the stalled
gaps' p99 (95 to 144 ms) toward the clean gaps' (40 to 74 ms). Most of Tokquay's ITL
excess over the baseline is not prefill: even the *clean* gaps at 6 and 8 requests/s
(median 52 and 62 ms) are far above the baseline's, because of the larger batches.
Chunked prefill pays when a prefill step is long compared with a decode step: long
prompts, or a model where prefill is compute-bound. GPT-2 small on this GPU with
prompts of at most 256 tokens is not that regime, and prompts longer than 256 were not
measured. (Extrapolating the two points above to GPT-2's 1,024-token limit gives a
stall of roughly 70 ms; that is a guess from two measurements, not a measurement.)

## Block size (16 vs 32)

Measured at the same KV budget in token slots (so 1,024 blocks of 16 or 512 of 32),
8 requests/s:

| KV budget | Block | Throughput | Mean KV utilisation | Preemptions |
|---|---:|---:|---:|---:|
| 16,384 | 16 | 782.5 tok/s | 96.7% | 16 |
| 16,384 | 32 | 764.6 | 93.5% | 36 |
| 8,192 | 16 | 726.1 | 96.7% | 203 |
| 8,192 | 32 | 728.5 | 93.5% | 180 |

Internal fragmentation is the measurable difference: doubling the block wasted twice
the slots (3.3% to 6.5% of allocated slots), as expected for about half a block lost
per sequence. Block 32's throughput was 2.3% lower at the larger budget and 0.3% higher
at the smaller, inside the run-to-run noise (Tokquay's identical 8 requests/s
configuration spread 2.6% across experiments). Preemption counts
went in opposite directions at the two budgets, so they are scheduling dynamics, not a
signal. The expected benefit of 32, fewer and larger gather chunks, did not show up:
the gather moves the same bytes whichever way they are cut into blocks. So 16 stays
the default; 32 costs memory and measured no gain here.
