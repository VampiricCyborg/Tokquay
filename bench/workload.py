"""Synthetic serving workload: Poisson arrivals, uniform prompt and output lengths.

The brief's workload: prompt lengths uniform in [16, 256], output lengths uniform in
[16, 256], requests arriving as a Poisson process. Requests always generate exactly
their output length (the benchmark sets ``ignore_eos``), so both systems do the same
work and the numbers do not depend on what GPT-2 happens to say.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

VOCAB_SIZE = 50257  # GPT-2; the last id is <|endoftext|>, which prompts avoid


@dataclass(frozen=True)
class RequestSpec:
    arrival: float  # seconds after the start of the run
    prompt_ids: tuple[int, ...]
    output_len: int


def make_workload(
    rate: float,
    n: int,
    seed: int = 0,
    prompt_len: tuple[int, int] = (16, 256),
    output_len: tuple[int, int] = (16, 256),
) -> list[RequestSpec]:
    """``n`` requests arriving as a Poisson process of ``rate`` requests per second
    (``rate=inf``: all at time 0, for measuring saturated throughput).

    Prompts and output lengths depend only on ``seed``, and inter-arrival gaps are
    drawn at unit rate and divided by ``rate``. So two workloads with the same seed
    contain the *same requests*, just compressed in time: runs at different rates are
    paired, and a difference between them is the load, not the luck of the draw.
    """
    if rate <= 0 or n < 1:
        raise ValueError("need rate > 0 and n >= 1")
    content, gaps = random.Random(seed), random.Random(seed + 1_000_003)
    t, out = 0.0, []
    for _ in range(n):
        t += gaps.expovariate(1.0)  # exponential gaps = a Poisson process
        prompt = tuple(content.randrange(VOCAB_SIZE - 1) for _ in range(content.randint(*prompt_len)))
        out.append(RequestSpec(t / rate, prompt, content.randint(*output_len)))
    return out
