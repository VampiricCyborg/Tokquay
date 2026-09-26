# Tokquay

**A from-scratch LLM inference engine with continuous batching and a paged KV cache.**
Project 5 of 10-projects-in-10-days.

Tokquay serves GPT-2 (124M) from a hand-written forward pass. It does not call `model.generate()`. The point is to build the two ideas that make vLLM fast and to measure them against a naive baseline:

1. **Paged KV cache:** KV memory is split into fixed-size blocks. Each sequence gets a block table, the same way an OS gives a process a page table. Nothing is allocated for the maximum length up front, so memory isn't fragmented and more sequences fit at once.
2. **Continuous (iteration-level) batching:** the scheduler admits and evicts requests at every decode step. A batch doesn't have to wait for its slowest member.

> What I'm learning: how inference memory is managed, how the scheduler trades throughput against latency, and why prefill and decode behave so differently.

---

## Scope (1 day, hard limits)

| In | Out |
|---|---|
| GPT-2 small, fp32, CPU or single GPU | Multi-GPU, tensor parallel |
| Own attention + KV cache in PyTorch | Custom CUDA/Triton kernels |
| Greedy + temperature/top-k sampling | Beam search, speculative decoding |
| Block allocator with preemption (recompute) | Swap-to-CPU preemption |
| FastAPI `/generate` with SSE streaming | OpenAI-compatible API, auth |
| Benchmark vs. static batching baseline | Prefix caching (stretch goal only) |

**Stack:** Python 3.11+, PyTorch, `transformers` (weights + tokenizer only), FastAPI, pytest.

---

## Architecture

```
            ┌──────────── FastAPI (/generate, SSE) ────────────┐
            │                                                  │
            ▼                                                  │
      ┌───────────┐   admit / preempt   ┌──────────────────┐   │
      │  Waiting  │ ──────────────────▶ │     Running      │   │
      │   queue   │ ◀────────────────── │  (≤ max_batch)   │   │
      └───────────┘   preempted (recompute)  └────────┬─────────┘   │
                                               │ step()        │
                                               ▼               │
                       ┌───────────────────────────────────┐   │
                       │ Engine.step()                     │   │
                       │  1. scheduler picks batch          │   │
                       │  2. allocator grows block tables   │   │
                       │  3. model forward (prefill+decode) │   │
                       │  4. sample, append, check stop     │───┘ stream tokens
                       └───────────────────────────────────┘
                                       │
                                       ▼
            ┌───────────────────── BlockAllocator ─────────────────────┐
            │ K/V pool: [num_blocks, n_layer, 2, n_head, BLOCK, d_head] │
            │ free list · ref counts · per-seq block_table: [int]       │
            └───────────────────────────────────────────────────────────┘
```

## Layout

```
tokquay/
  model.py        # GPT-2 forward pass written by hand; attention reads KV via block tables
  kv_cache.py     # BlockAllocator: pool tensor, free list, alloc/free/append_slot
  sequence.py     # Sequence state: prompt ids, output ids, block_table, status
  scheduler.py    # FCFS continuous batching, token budget, preemption
  engine.py       # step() loop, sampling, stop conditions
  server.py       # FastAPI + SSE streaming, background engine loop
  baseline.py     # static batching with contiguous cache, used for comparison
bench/
  bench.py        # Poisson arrivals, varied prompt/output lengths
tests/
```

---

## Build phases

Each phase must pass its tests before the next one starts.

### Phase 1: Hand-written GPT-2 forward (≈1.5h)
- Load HF GPT-2 weights into your own `nn.Module`: embeddings, 12 blocks, LN, attention, MLP, tied LM head.
- No cache yet. Run the full sequence every step.
- **Test:** logits match `GPT2LMHeadModel` to `atol=1e-4` on 5 prompts. 20 greedy tokens match HF `generate(do_sample=False)` exactly.

### Phase 2: Contiguous KV cache (≈1h)
- Split the forward into prefill (the whole prompt) and decode (one token that attends to the cached K/V).
- **Test:** greedy output is identical to Phase 1. Decode gets faster per token as length grows (log it).

### Phase 3: Paged KV cache (≈2.5h) ← the core
- `BlockAllocator` holds one preallocated pool tensor, a free list, and `allocate(seq)`, `append_slot(seq)` (grabs a new block when the current one fills) and `free(seq)`.
- Attention gathers K/V from `pool[block_table]`, masks the unused slots in the last block, and runs one batched attention over sequences of different lengths.
- **Tests:**
  - Output is identical to Phase 2 for a single sequence.
  - The free-block count returns to its initial value after any sequence of alloc/free calls (property test with `hypothesis`).
  - 8 sequences of different lengths in one batch produce the same output as running each alone.

### Phase 4: Continuous batching scheduler (≈2h)
- FCFS waiting queue. Every step, admit new sequences while blocks and the `max_num_batched_tokens` budget allow.
- Mix prefill and decode in one step (or alternate prefill-only and decode-only steps; document which you chose and why).
- Preemption: when blocks run out, evict the newest running sequence, free its blocks, and put it back in the waiting queue so it is recomputed later.
- **Tests:**
  - Requests that arrive mid-generation get admitted without waiting for the batch to drain.
  - Forcing a tiny pool (for example 16 blocks) triggers preemption, and every request still finishes with correct output.
  - No deadlock when one request needs more blocks than the pool has; reject it up front.

### Phase 5: Server + streaming (≈1h)
- The engine runs in a background thread or asyncio task. Each request gets an `asyncio.Queue` for its tokens.
- `POST /generate {prompt, max_tokens, temperature, top_k, stream}` streams tokens over SSE.
- **Test:** 20 concurrent clients all get complete, correct outputs.

### Phase 6: Benchmark + README numbers (≈1h)
- `baseline.py`: static batches of N with a contiguous cache sized to `max_len`. No request joins until the batch finishes.
- Workload: Poisson arrivals at 3 rates. Prompt lengths are uniform in [16, 256] and output lengths in [16, 256].
- Report: throughput (tokens/s), p50/p99 time-to-first-token, p50/p99 inter-token latency, peak KV memory, and the largest concurrent batch reached.
- Plot throughput vs. arrival rate for Tokquay and the baseline.

---

## Results

Only real numbers from `bench/bench.py` (RTX 3050 Laptop 4 GiB, fp32, 300 requests, Poisson arrivals at 4 requests/s,
prompt and output lengths uniform in [16, 256], both systems given the same 1.125 GiB of KV memory). The other arrival
rates, the plots and the caveats are in the README and `Documentation/results.md`.

| Metric | Static baseline | Tokquay |
|---|---|---|
| Throughput (tok/s) | 544.8 | 570.9 |
| TTFT p50 / p99 (ms) | 2,901 / 6,400 | 38 / 74 |
| ITL p50 / p99 (ms) | 21.8 / 42.0 | 26.6 / 74.7 |
| Peak KV memory in use (MiB) | 1152 | 568 |
| Max concurrent seqs | 32 | 32 |

## Design decisions to write up
Written up with their measurements in `design-notes.md`.
- Block size (16 vs 32): internal fragmentation vs. gather overhead. Measure both.
- Recompute vs. swap for preemption, and why recompute is fine at this scale.
- Chunked prefill: not implemented. Explain what problem it solves (long prompts stalling decode).

## Stretch (only if everything above is done)
- Prefix caching: hash full blocks, share them via ref counts, copy-on-write on append.

## References
- Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention* (SOSP '23)
- Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative Models* (OSDI '22)
