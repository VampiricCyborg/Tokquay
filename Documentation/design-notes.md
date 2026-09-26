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
