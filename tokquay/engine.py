"""Phase 4: the engine ties model + allocator + scheduler into a step loop.

``Engine.step()`` is one iteration of continuous batching:

  1. the scheduler picks a batch (admitting / preempting as needed);
  2. blocks were already grown by the scheduler for the tokens about to be written;
  3. one paged forward over the batch (prefill *or* decode, see scheduler.py);
  4. sample one token per sequence, append it, and check stop conditions.

It is synchronous and not thread-safe; the server (Phase 5, ``async_engine.py``)
drives it from one dedicated thread and turns the returned ``TokenEvent``s into
streamed responses.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tokquay.generate import paged_forward
from tokquay.kv_cache import BlockAllocator
from tokquay.model import GPT2
from tokquay.scheduler import Scheduler, SchedulerConfig
from tokquay.sequence import SamplingParams, Sequence

GPT2_EOS_TOKEN_ID = 50256


@dataclass
class TokenEvent:
    """One newly generated token. ``finish_reason`` is set on a sequence's last token."""

    seq_id: int
    token_id: int
    finish_reason: str | None = None  # "length" | "stop"


@dataclass
class EngineStats:
    steps: int = 0
    prefill_steps: int = 0
    decode_steps: int = 0
    tokens_generated: int = 0
    max_batch: int = 0  # most sequences in one step
    peak_used_blocks: int = 0  # high-water mark of KV blocks in use
    num_preemptions: int = 0
    # Work accounting, in token positions run through the model:
    positions_computed: int = 0  # real positions fed (prompt + recomputed + decoded)
    positions_padded: int = 0  # what the model actually computed: batch x longest row
    positions_discarded: int = 0  # cached positions thrown away by preemption (all recomputed)


@dataclass
class KVSnapshot:
    """KV memory right now, in token slots (multiply by bytes per token for bytes)."""

    allocated_tokens: int  # slots held by running sequences (blocks x block size; or rows x max_len)
    live_tokens: int  # slots that actually hold a cached token
    num_seqs: int  # sequences holding KV memory


@dataclass
class StepInfo:
    """What the last ``step()`` ran. Shared by ``Engine`` and the baseline so the
    benchmark can drive either one."""

    is_prefill: bool
    num_seqs: int  # sequences in the step
    padded_tokens: int  # batch size x longest row
    kv: KVSnapshot  # KV memory at the step's peak: after the forward, before finished sequences are freed


def finish_reason_for(seq: Sequence, token: int, eos_token_id: int) -> str | None:
    """Why ``seq`` is done after sampling ``token`` (None if it is not)."""
    if not seq.sampling.ignore_eos and token == eos_token_id:
        return "stop"
    if len(seq.output_token_ids) >= seq.sampling.max_tokens:
        return "length"
    return None


def sample_token(logits: torch.Tensor, params: SamplingParams, generator: torch.Generator | None) -> int:
    """Sample from one row of logits ``[vocab]``: greedy, or temperature + optional top-k."""
    if params.temperature == 0:
        return int(logits.argmax())
    logits = logits / params.temperature
    if params.top_k > 0:
        kth = torch.topk(logits, min(params.top_k, logits.numel())).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator))


class Engine:
    def __init__(
        self,
        model: GPT2,
        allocator: BlockAllocator,
        config: SchedulerConfig | None = None,
        eos_token_id: int = GPT2_EOS_TOKEN_ID,
    ):
        if config is None:
            config = SchedulerConfig(max_model_len=model.cfg.n_positions)
        if config.max_model_len > model.cfg.n_positions:
            raise ValueError("max_model_len exceeds the model's context window")
        self.model = model
        self.allocator = allocator
        self.scheduler = Scheduler(allocator, config)
        self.eos_token_id = eos_token_id
        self.stats = EngineStats()
        self.last_step: StepInfo | None = None  # None if the last step() had nothing to run
        self._next_id = 0
        self._rngs: dict[int, torch.Generator] = {}

    # ---- requests ------------------------------------------------------------------
    def add_request(self, prompt_token_ids: list[int], params: SamplingParams | None = None) -> Sequence:
        """Queue a request. Raises ``RequestRejected`` if it can never be served."""
        seq = Sequence(self._next_id, list(prompt_token_ids), sampling=params or SamplingParams())
        self.scheduler.add_request(seq)  # may raise; the id is only consumed on success
        self._next_id += 1
        if seq.sampling.temperature > 0:
            device = self.allocator.pool.device
            gen = torch.Generator(device=device)
            gen.manual_seed(seq.sampling.seed if seq.sampling.seed is not None else torch.seed() % 2**63)
            self._rngs[seq.seq_id] = gen
        return seq

    def abort_request(self, seq: Sequence) -> None:
        """Cancel a request (e.g. its client disconnected) and free its KV blocks.

        Call between steps, never while ``step()`` is running."""
        self.scheduler.abort(seq)
        self._rngs.pop(seq.seq_id, None)

    def has_unfinished(self) -> bool:
        return self.scheduler.has_unfinished()

    def kv_snapshot(self) -> KVSnapshot:
        running = self.scheduler.running
        return KVSnapshot(
            allocated_tokens=self.allocator.num_used_blocks * self.allocator.block_size,
            live_tokens=sum(s.num_cached_tokens for s in running),
            num_seqs=len(running),
        )

    # ---- the loop ------------------------------------------------------------------
    @torch.no_grad()
    def step(self) -> list[TokenEvent]:
        """Run one iteration; returns the tokens produced (empty if idle)."""
        batch = self.scheduler.schedule()
        if batch is None:
            self.last_step = None
            return []

        stats = self.stats
        stats.steps += 1
        stats.prefill_steps += batch.is_prefill
        stats.decode_steps += not batch.is_prefill
        stats.max_batch = max(stats.max_batch, len(batch.seqs))
        stats.peak_used_blocks = max(stats.peak_used_blocks, self.allocator.num_used_blocks)
        stats.num_preemptions = self.scheduler.num_preemptions
        stats.positions_discarded = self.scheduler.num_discarded_positions
        stats.positions_computed += sum(s.num_uncached_tokens for s in batch.seqs)  # before the forward advances the cache
        stats.positions_padded += batch.padded_tokens

        logits = paged_forward(self.model, self.allocator, batch.seqs)  # [B, vocab]
        self.last_step = StepInfo(batch.is_prefill, len(batch.seqs), batch.padded_tokens, self.kv_snapshot())
        tokens = self._sample(batch.seqs, logits)

        events = []
        for seq, tok in zip(batch.seqs, tokens):
            seq.append_token(tok)
            reason = self._finish_reason(seq, tok)
            if reason:
                self.scheduler.finish(seq, reason)
                self._rngs.pop(seq.seq_id, None)
            events.append(TokenEvent(seq.seq_id, tok, reason))
        stats.tokens_generated += len(events)
        return events

    def run_until_done(self) -> list[TokenEvent]:
        events = []
        while self.has_unfinished():
            events.extend(self.step())
        return events

    # ---- helpers -------------------------------------------------------------------
    def _sample(self, seqs: list[Sequence], logits: torch.Tensor) -> list[int]:
        tokens = logits.argmax(-1)  # greedy rows in one batched op
        sampled = [i for i, s in enumerate(seqs) if s.sampling.temperature > 0]
        for i in sampled:
            tokens[i] = sample_token(logits[i], seqs[i].sampling, self._rngs[seqs[i].seq_id])
        return tokens.tolist()  # the step's single GPU sync

    def _finish_reason(self, seq: Sequence, token: int) -> str | None:
        return finish_reason_for(seq, token, self.eos_token_id)
