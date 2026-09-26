"""Per-request state that the allocator, scheduler and engine all share."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class SequenceStatus(enum.Enum):
    WAITING = "waiting"  # queued (or preempted and waiting to be recomputed)
    RUNNING = "running"  # owns KV blocks and is part of the batch
    FINISHED = "finished"


@dataclass(frozen=True)
class SamplingParams:
    """Per-request generation settings. ``temperature == 0`` means greedy."""

    max_tokens: int = 16
    temperature: float = 0.0
    top_k: int = 0  # 0 disables top-k filtering
    ignore_eos: bool = False  # keep generating past EOS (benchmarks want fixed lengths)
    seed: int | None = None  # per-request sampling seed; None = nondeterministic

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")


@dataclass
class Sequence:
    """One generation request.

    ``num_cached_tokens`` is how many leading positions already have their K/V
    written to the pool. The tokens still to be fed to the model on the next
    step are therefore ``token_ids[num_cached_tokens:]``:

    * fresh prompt (prefill): the whole prompt;
    * steady state (decode): just the last sampled token;
    * after preemption by recompute: prompt + everything generated so far.

    ``block_table[i]`` is the pool block holding positions ``[i*B, (i+1)*B)``.
    """

    seq_id: int
    prompt_token_ids: list[int]
    output_token_ids: list[int] = field(default_factory=list)
    block_table: list[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING
    num_cached_tokens: int = 0
    sampling: SamplingParams = field(default_factory=SamplingParams)
    finish_reason: str | None = None  # "length", "stop" or "abort" once finished

    def __post_init__(self) -> None:
        if not self.prompt_token_ids:
            raise ValueError("prompt must contain at least one token")

    @property
    def token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.output_token_ids

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def num_uncached_tokens(self) -> int:
        return self.num_tokens - self.num_cached_tokens

    def append_token(self, token_id: int) -> None:
        self.output_token_ids.append(token_id)

    def reset_for_recompute(self) -> None:
        """Forget cache state after the allocator freed our blocks (preemption)."""
        self.block_table = []
        self.num_cached_tokens = 0
        self.status = SequenceStatus.WAITING
