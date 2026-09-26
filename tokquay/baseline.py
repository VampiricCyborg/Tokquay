"""Phase 6: the static-batching baseline Tokquay is measured against.

This is what serving looked like before continuous batching and paging, built as
fairly as it can be with the same model, dtype and kernels-level code path:

* **Static batches.** When the engine is idle it takes up to ``batch_size`` waiting
  requests (FCFS), prefills them together, and decodes them together until the
  *slowest* one finishes. No request joins while a batch is running, so a request
  that arrives one step after a batch started waits for all of it to drain.
* **Rows stay occupied.** A row that finishes early keeps its slot and is fed a dummy
  token (its output is discarded) until the whole batch is done. That is how a
  batched ``generate()`` behaves: the batch tensor keeps its shape.
* **Contiguous, reserved KV cache.** Every row owns ``max_len`` slots up front,
  whether or not the request will ever need them. Nothing is allocated on demand,
  which is exactly the memory waste paging removes.
* **No paging overhead.** Attention reads each row's K/V with a plain slice
  (``ContiguousBatch.gather``), not a block-table gather, so the baseline is not
  taxed for a mechanism it does not use.

It has the same driving interface as ``Engine`` (``add_request`` / ``step`` /
``has_unfinished`` / ``kv_snapshot`` / ``last_step``) so the benchmark can run either.
Greedy decoding only: that is all the benchmark uses.
"""

from __future__ import annotations

from collections import deque

import torch

from tokquay.engine import (
    GPT2_EOS_TOKEN_ID,
    EngineStats,
    KVSnapshot,
    StepInfo,
    TokenEvent,
    finish_reason_for,
)
from tokquay.kv_cache import BatchedKV
from tokquay.model import GPT2
from tokquay.scheduler import RequestRejected
from tokquay.sequence import SamplingParams, Sequence, SequenceStatus


class ContiguousBatch(BatchedKV):
    """``PagedBatch``'s twin for a static batch: row ``b`` owns ``k[:, b]`` / ``v[:, b]``
    outright (``[n_layer, B, H, max_len, d]``), and position ``p`` of row ``b`` lives at
    slot ``p``. Same per-row ``start`` / ``new_lens`` contract and the same mask rule."""

    def __init__(self, k: torch.Tensor, v: torch.Tensor, start: list[int], new_lens: list[int]):
        if not start or len(start) != len(new_lens) or len(start) > k.shape[1] or min(new_lens) < 1:
            raise ValueError("need 1..batch_size rows, each feeding at least one token")
        ctx = [s + n for s, n in zip(start, new_lens)]
        if max(ctx) > k.shape[3]:
            raise ValueError(f"a row needs {max(ctx)} slots but max_len is {k.shape[3]}")
        device = k.device
        self.k, self.v = k, v
        self.B, self.T, self.max_ctx = len(start), max(new_lens), max(ctx)

        start_t = torch.tensor(start, dtype=torch.long, device=device)
        ctx_t = torch.tensor(ctx, dtype=torch.long, device=device)
        self.positions = start_t[:, None] + torch.arange(self.T, device=device)[None, :]
        self.last_idx = torch.tensor([n - 1 for n in new_lens], dtype=torch.long, device=device)
        k_pos = torch.arange(self.max_ctx, device=device)[None, None, :]
        self.mask = ((k_pos <= self.positions[:, :, None]) & (k_pos < ctx_t[:, None, None]))[:, None]

        # Scatter targets for the valid new tokens only: (row, token column, slot).
        w_b, w_t, w_pos = [], [], []
        for b, (s, n) in enumerate(zip(start, new_lens)):
            w_b += [b] * n
            w_t += range(n)
            w_pos += range(s, s + n)
        self._w_b, self._w_t, self._w_pos = torch.tensor([w_b, w_t, w_pos], dtype=torch.long, device=device)

    def write(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> None:
        self.k[layer][self._w_b, :, self._w_pos] = k[self._w_b, :, self._w_t]  # [N, H, d]
        self.v[layer][self._w_b, :, self._w_pos] = v[self._w_b, :, self._w_t]

    def gather(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.max_ctx  # a slice (a view), not a gather: this is the contiguous layout's advantage
        return self.k[layer, : self.B, :, :s], self.v[layer, : self.B, :, :s]


class StaticBatchEngine:
    def __init__(
        self,
        model: GPT2,
        batch_size: int,
        max_len: int,
        device=None,
        eos_token_id: int = GPT2_EOS_TOKEN_ID,
    ):
        cfg = model.cfg
        if batch_size < 1 or not 1 <= max_len <= cfg.n_positions:
            raise ValueError("need batch_size >= 1 and 1 <= max_len <= the model's context window")
        param = next(model.parameters())
        shape = (cfg.n_layer, batch_size, cfg.n_head, max_len, cfg.d_head)
        self.k = torch.zeros(shape, device=device or param.device, dtype=param.dtype)
        self.v = torch.zeros_like(self.k)  # the whole reservation exists from the start
        self.model, self.batch_size, self.max_len, self.eos_token_id = model, batch_size, max_len, eos_token_id
        self.bytes_per_token = 2 * cfg.n_layer * cfg.n_head * cfg.d_head * self.k.element_size()
        self.waiting: deque[Sequence] = deque()
        self.batch: list[Sequence] = []  # the static batch in flight; empty between batches
        self.stats = EngineStats()
        self.last_step: StepInfo | None = None
        self.num_batches = 0
        self._next_id = 0

    # ---- requests ------------------------------------------------------------------
    def add_request(self, prompt_token_ids: list[int], params: SamplingParams | None = None) -> Sequence:
        params = params or SamplingParams()
        if params.temperature != 0:
            raise ValueError("the static baseline only does greedy decoding")
        seq = Sequence(self._next_id, list(prompt_token_ids), sampling=params)
        if len(seq.prompt_token_ids) + params.max_tokens > self.max_len:
            # Strictly one more than the engine's rule: a finished row keeps writing a dummy
            # token into the next slot until its batch ends, so that slot must exist.
            raise RequestRejected(f"prompt + max_tokens exceeds the cache's max_len ({self.max_len})")
        self._next_id += 1
        self.waiting.append(seq)
        return seq

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.batch)

    def kv_snapshot(self) -> KVSnapshot:
        rows = len(self.batch)  # finished rows still hold their slab until the batch ends
        return KVSnapshot(
            allocated_tokens=rows * self.max_len,
            live_tokens=sum(s.num_cached_tokens for s in self.batch),
            num_seqs=rows,
        )

    # ---- the loop ------------------------------------------------------------------
    @torch.no_grad()
    def step(self) -> list[TokenEvent]:
        """One forward pass: the prefill of a new batch, or one decode step of the current one."""
        if not self.batch:
            if not self.waiting:
                self.last_step = None
                return []
            self.batch = [self.waiting.popleft() for _ in range(min(self.batch_size, len(self.waiting)))]
            for s in self.batch:
                s.status = SequenceStatus.RUNNING
            self.num_batches += 1
            is_prefill = True
            feeds = [s.prompt_token_ids for s in self.batch]
            starts = [0] * len(self.batch)
        else:
            is_prefill = False
            # Every row is fed: finished ones get a dummy token whose output is discarded.
            feeds = [[s.output_token_ids[-1]] if s.status is SequenceStatus.RUNNING else [0] for s in self.batch]
            starts = [s.num_cached_tokens for s in self.batch]

        view = ContiguousBatch(self.k, self.v, starts, [len(f) for f in feeds])
        ids = torch.tensor([f + [0] * (view.T - len(f)) for f in feeds], dtype=torch.long, device=self.k.device)
        logits = self.model(ids, view, last_only=True)[:, 0]
        tokens = logits.argmax(-1).tolist()  # the step's single GPU sync

        st = self.stats
        st.steps += 1
        st.prefill_steps += is_prefill
        st.decode_steps += not is_prefill
        st.max_batch = max(st.max_batch, len(self.batch))
        st.positions_padded += view.B * view.T

        events = []
        for seq, feed, tok in zip(self.batch, feeds, tokens):
            if seq.status is not SequenceStatus.RUNNING:
                continue  # a finished row: computed, discarded
            st.positions_computed += len(feed)
            seq.num_cached_tokens += len(feed)
            seq.append_token(tok)
            reason = finish_reason_for(seq, tok, self.eos_token_id)
            if reason:
                seq.status, seq.finish_reason = SequenceStatus.FINISHED, reason
            events.append(TokenEvent(seq.seq_id, tok, reason))
        st.tokens_generated += len(events)
        self.last_step = StepInfo(is_prefill, len(self.batch), view.B * view.T, self.kv_snapshot())
        if all(s.status is SequenceStatus.FINISHED for s in self.batch):
            self.batch = []  # the slowest row is done: only now can the next batch form
        return events

    def run_until_done(self) -> list[TokenEvent]:
        events = []
        while self.has_unfinished():
            events.extend(self.step())
        return events
