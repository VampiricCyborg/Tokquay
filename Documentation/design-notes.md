# Design notes

Decisions made while building Tokquay, with the reasoning. The brief lists three
write-ups still owed (block size, recompute vs. swap, chunked prefill); the ones
that are already settled are filled in here, the measured ones are marked TODO
until Phase 6 produces real numbers.

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
  is that at GPT-2 124M scale re-prefilling a few hundred tokens is cheap
  enough that this is the right trade; that is unmeasured so far.
  TODO(Phase 6): report how many preemptions the benchmark caused and how much
  of the forward-pass work was recomputation, to confirm or refute it.

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

## Chunked prefill (not implemented)

A long prompt is prefilled in one step, and the whole running batch waits for
it. Chunked prefill splits a long prompt into pieces of at most N tokens and
spreads them over several steps, mixed with decodes, so no step is long and
decode latency stays flat. It needs ragged attention (or per-row padding
tricks) to avoid the padding waste described above. TODO(Phase 6): compare the
p99 inter-token latency we see with prompt length to quantify what it would buy.

## Block size (16 vs 32)

TODO(Phase 6): measure both. Expectation: a larger block wastes more slots in
each sequence's last block (internal fragmentation, up to `block_size - 1`
tokens per sequence) but gathers fewer, larger chunks per step.
