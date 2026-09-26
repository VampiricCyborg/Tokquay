"""Phase 4: FCFS continuous (iteration-level) batching scheduler.

The scheduler decides, at *every* step, which sequences run. Requests join the
running batch the moment blocks and budget allow and leave the moment they
finish; nothing waits for a batch to drain.

Policy choices (see also Documentation/design-notes.md)
-------------------------------------------------------
* **Alternating, prefill-priority steps** (not mixed). If any waiting request can
  be admitted, the step is a prefill-only step over the newly admitted requests;
  otherwise it is a decode-only step over all running sequences. Mixed steps are
  possible in the model (per-sequence start/length), but our batches are padded
  to the longest row, so putting one long prompt next to N one-token decode rows
  would make every decode row compute a prompt's worth of padding. Alternating
  keeps decode steps at exactly T=1. The price is that running sequences stall
  for the duration of a prefill step (a latency spike that chunked prefill
  would smooth out; not implemented).
* **FCFS admission with no skipping.** Admission stops at the first waiting
  request that does not fit, so a large request cannot be starved by a stream
  of small ones.
* **Preemption by recompute.** When a decode step cannot get a block for some
  sequence, the *newest* running sequence is evicted: its blocks are freed and
  it goes to the *front* of the waiting queue (keeping arrival order). It keeps
  its generated tokens; when re-admitted, prompt + output are prefilled again.
* **Up-front rejection.** A request whose worst-case context can never fit
  (pool, model length, or per-step token budget) is refused at ``add_request``.
  Together with "the oldest running sequence is never the victim unless it is
  alone", this guarantees progress: no deadlock, no livelock.
* **Watermark.** While other sequences are running, admission leaves
  ``watermark_blocks`` free so a just-admitted sequence is not immediately
  evicted by its neighbours' next block boundary. A lone request ignores it.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from tokquay.kv_cache import BlockAllocator
from tokquay.sequence import Sequence, SequenceStatus


class RequestRejected(ValueError):
    """The request can never be served by this engine configuration."""


@dataclass
class SchedulerConfig:
    max_num_seqs: int = 64  # max sequences running at once
    # Budget for one step, counted in *padded* tokens (batch size x longest row), since
    # that is what the model actually computes and what bounds activation memory.
    max_num_batched_tokens: int = 2048
    max_model_len: int = 1024  # GPT-2's context window
    watermark_blocks: int = 1

    def __post_init__(self) -> None:
        if self.max_num_seqs < 1 or self.max_num_batched_tokens < self.max_num_seqs:
            raise ValueError("need max_num_seqs >= 1 and max_num_batched_tokens >= max_num_seqs")


@dataclass
class ScheduledBatch:
    seqs: list[Sequence]
    is_prefill: bool
    preempted: list[Sequence] = field(default_factory=list)  # evicted while building this batch
    padded_tokens: int = field(init=False)  # batch size x longest row: what the model computes

    def __post_init__(self) -> None:
        # Fixed at creation: num_uncached_tokens changes once the step has run.
        self.padded_tokens = len(self.seqs) * max(s.num_uncached_tokens for s in self.seqs)


class Scheduler:
    def __init__(self, allocator: BlockAllocator, config: SchedulerConfig | None = None):
        self.allocator = allocator
        self.config = config or SchedulerConfig()
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []  # in admission order: the last one is the newest
        self.num_preemptions = 0

    # ---- requests ------------------------------------------------------------------
    def max_context(self, seq: Sequence) -> int:
        """Most positions this sequence will ever have in the cache (the last sampled
        token is returned to the caller and never needs K/V)."""
        return len(seq.prompt_token_ids) + seq.sampling.max_tokens - 1

    def rejection_reason(self, seq: Sequence) -> str | None:
        ctx, cfg = self.max_context(seq), self.config
        if ctx > cfg.max_model_len:
            return f"prompt + max_tokens needs {ctx} positions but the model supports {cfg.max_model_len}"
        blocks = self.allocator.blocks_needed(ctx)
        if blocks > self.allocator.num_blocks:
            return f"needs {blocks} KV blocks but the pool only has {self.allocator.num_blocks}"
        if ctx > cfg.max_num_batched_tokens:
            # A preempted sequence is re-prefilled whole; it must fit in one step's budget.
            return f"needs {ctx} tokens in one step but max_num_batched_tokens is {cfg.max_num_batched_tokens}"
        return None

    def add_request(self, seq: Sequence) -> None:
        reason = self.rejection_reason(seq)
        if reason:
            raise RequestRejected(reason)
        seq.status = SequenceStatus.WAITING
        self.waiting.append(seq)

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    def finish(self, seq: Sequence, reason: str) -> None:
        self.running.remove(seq)
        self.allocator.free(seq)
        seq.status = SequenceStatus.FINISHED
        seq.finish_reason = reason

    # ---- one scheduling decision ---------------------------------------------------
    def schedule(self) -> ScheduledBatch | None:
        """Pick the sequences for the next step (or None if there is nothing to do)."""
        if self.waiting:
            admitted = self._admit()
            if admitted:
                self.running.extend(admitted)
                return ScheduledBatch(admitted, is_prefill=True)
        if not self.running:
            return None
        return self._schedule_decode()

    def _admit(self) -> list[Sequence]:
        cfg, alloc = self.config, self.allocator
        admitted: list[Sequence] = []
        longest = 0
        while self.waiting and len(self.running) + len(admitted) < cfg.max_num_seqs:
            seq = self.waiting[0]
            n = seq.num_tokens  # everything is uncached: prompt (+ output, if recomputing)
            if (len(admitted) + 1) * max(longest, n) > cfg.max_num_batched_tokens:
                break
            others_running = bool(self.running or admitted)
            reserve = cfg.watermark_blocks if others_running else 0
            if alloc.blocks_needed(n) + reserve > alloc.num_free_blocks:
                break  # FCFS: never skip ahead of the head of the queue
            self.waiting.popleft()
            alloc.allocate(seq)
            seq.status = SequenceStatus.RUNNING
            longest = max(longest, n)
            admitted.append(seq)
        return admitted

    def _schedule_decode(self) -> ScheduledBatch:
        """Make room for every running sequence's next token, evicting the newest
        sequences when the pool is exhausted."""
        alloc, running = self.allocator, self.running
        preempted: list[Sequence] = []
        i = 0
        while i < len(running):
            seq = running[i]
            while not alloc.can_append_slot(seq):
                victim = running.pop()  # newest; may be `seq` itself if it is the last one
                self._preempt(victim)
                preempted.append(victim)
                if victim is seq:
                    break
            else:
                alloc.append_slot(seq)
                i += 1
        # The oldest sequence is only ever evicted if it is alone, and then the pool is
        # entirely free for it (rejection guarantees it fits), so this cannot be empty.
        assert running, "scheduler evicted every running sequence"
        return ScheduledBatch(list(running), is_prefill=False, preempted=preempted)

    def _preempt(self, seq: Sequence) -> None:
        self.allocator.free(seq)
        seq.reset_for_recompute()  # keeps output_token_ids; status -> WAITING
        self.waiting.appendleft(seq)  # front: keeps arrival order among waiting requests
        self.num_preemptions += 1
