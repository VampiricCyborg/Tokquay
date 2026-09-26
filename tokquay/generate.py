"""Simple single-sequence greedy generation, with and without a KV cache.

Used for Phase 1/2 verification and timing; the real serving path is the engine.
"""

from __future__ import annotations

import time

import torch

from tokquay.kv_cache import BlockAllocator
from tokquay.model import GPT2, ContiguousKVCache
from tokquay.sequence import Sequence, SequenceStatus


@torch.no_grad()
def greedy_generate(
    model: GPT2,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    use_cache: bool = True,
    step_times: list[float] | None = None,
) -> list[int]:
    """Greedy decode for a batch of one. If ``step_times`` is given, append the
    wall-clock seconds of each decode step (prefill excluded when caching)."""
    device = input_ids.device
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    out: list[int] = []

    if use_cache:
        cache = ContiguousKVCache(
            model.cfg,
            batch_size=1,
            max_len=input_ids.shape[1] + max_new_tokens,
            device=device,
            dtype=next(model.parameters()).dtype,
        )
        logits = model(input_ids, cache, last_only=True)  # prefill
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        out.append(nxt.item())
        for _ in range(max_new_tokens - 1):
            sync()
            t0 = time.perf_counter()
            logits = model(nxt, cache, last_only=True)  # decode: one token
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            out.append(nxt.item())  # .item() syncs
            if step_times is not None:
                step_times.append(time.perf_counter() - t0)
    else:
        cur = input_ids
        for _ in range(max_new_tokens):
            sync()
            t0 = time.perf_counter()
            nxt = model(cur, last_only=True)[:, -1].argmax(-1, keepdim=True)
            cur = torch.cat([cur, nxt], dim=1)
            out.append(nxt.item())
            if step_times is not None:
                step_times.append(time.perf_counter() - t0)
    return out


@torch.no_grad()
def paged_forward(
    model: GPT2,
    allocator: BlockAllocator,
    seqs: list[Sequence],
    new_lens: list[int] | None = None,
) -> torch.Tensor:
    """Run one batched step over paged KV and return next-token logits ``[B, vocab]``.

    Each sequence feeds ``token_ids[num_cached_tokens : num_cached_tokens + n]``
    (``n`` defaults to everything uncached), so a batch may freely mix prefill
    (a whole prompt) and decode (one token). Blocks must already cover the
    sequences (``allocate`` / ``append_slot``). Advances ``num_cached_tokens``.
    """
    lens = new_lens or [s.num_uncached_tokens for s in seqs]
    batch = allocator.batch_for(seqs, lens)
    rows = []
    for s, n in zip(seqs, lens):
        row = s.token_ids[s.num_cached_tokens : s.num_cached_tokens + n]
        rows.append(row + [0] * (batch.T - n))  # right-pad; padded slots never write to the pool
    ids = torch.tensor(rows, dtype=torch.long, device=allocator.pool.device)
    logits = model(ids, batch, last_only=True)[:, 0]
    for s, n in zip(seqs, lens):
        s.num_cached_tokens += n
    return logits


@torch.no_grad()
def paged_greedy_generate(
    model: GPT2,
    allocator: BlockAllocator,
    prompts: list[list[int]],
    max_new_tokens: int,
) -> list[list[int]]:
    """Greedy-decode several prompts together on the paged cache.

    A minimal driver (no scheduler): all prompts are prefilled in one padded
    batch, then decoded together; a sequence's blocks are freed the moment it
    finishes. The real serving loop with admission and preemption is Phase 4.
    """
    seqs = [Sequence(i, list(p)) for i, p in enumerate(prompts)]
    for s in seqs:
        allocator.allocate(s)
        s.status = SequenceStatus.RUNNING
    running = seqs
    while running:
        next_ids = paged_forward(model, allocator, running).argmax(-1).tolist()
        still_running = []
        for s, t in zip(running, next_ids):
            s.append_token(t)
            if len(s.output_token_ids) >= max_new_tokens:
                allocator.free(s)
                s.status = SequenceStatus.FINISHED
            else:
                allocator.append_slot(s)  # room for the token we just sampled
                still_running.append(s)
        running = still_running
    return [s.output_token_ids for s in seqs]
