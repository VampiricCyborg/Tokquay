"""Simple single-sequence greedy generation, with and without a KV cache.

Used for Phase 1/2 verification and timing; the real serving path is the engine.
"""

from __future__ import annotations

import time

import torch

from tokquay.model import GPT2, ContiguousKVCache


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
