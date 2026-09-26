"""Drive a serving system in real time and turn what happened into the brief's metrics.

``run_trace`` works on anything with the ``Engine`` interface (``add_request``,
``step``, ``has_unfinished``, ``last_step``): the Tokquay engine and
the static baseline. It is an *open loop*: requests arrive when the workload says,
whether or not the system has kept up. Latency is measured from the scheduled arrival,
so time a request spends waiting because the driver was busy inside a step counts
against the system, as it would for a real client.

A token's timestamp is the moment the step that produced it returned (the step ends
with a GPU sync, so this is when the token could first have been sent).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np

from bench.workload import RequestSpec
from tokquay.sequence import SamplingParams


@dataclass
class RequestRecord:
    arrival: float
    prompt_len: int
    output_len: int
    token_times: list[float] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)


@dataclass
class StepRecord:
    t_start: float
    t_end: float
    is_prefill: bool
    num_seqs: int  # sequences in the step
    padded_tokens: int  # batch size x longest row
    allocated_tokens: int  # KV slots held after the step
    live_tokens: int  # of which hold a cached token
    num_running: int  # sequences holding KV memory


@dataclass
class RunResult:
    requests: list[RequestRecord]
    steps: list[StepRecord]
    stats: dict  # the system's own counters (preemptions, positions computed, ...)
    wall_s: float


def run_trace(
    system,
    workload: list[RequestSpec],
    *,
    clock=time.perf_counter,
    sleep=time.sleep,
    idle_quantum: float = 0.005,
) -> RunResult:
    records: dict[int, RequestRecord] = {}
    steps: list[StepRecord] = []
    i, n = 0, len(workload)
    t0 = clock()
    while i < n or system.has_unfinished():
        now = clock() - t0
        while i < n and workload[i].arrival <= now:
            spec, i = workload[i], i + 1
            seq = system.add_request(list(spec.prompt_ids), SamplingParams(max_tokens=spec.output_len, ignore_eos=True))
            records[seq.seq_id] = RequestRecord(spec.arrival, len(spec.prompt_ids), spec.output_len)
        if not system.has_unfinished():
            sleep(max(0.0, min(workload[i].arrival - now, idle_quantum)))  # idle until the next arrival
            continue

        start = clock() - t0
        events = system.step()
        end = clock() - t0
        info = system.last_step
        if info is None:
            raise RuntimeError("the system has unfinished requests but nothing it can run (deadlock)")
        kv = info.kv
        steps.append(
            StepRecord(
                start, end, info.is_prefill, info.num_seqs, info.padded_tokens,
                kv.allocated_tokens, kv.live_tokens, kv.num_seqs,
            )
        )
        for ev in events:
            rec = records[ev.seq_id]
            rec.token_times.append(end)
            rec.token_ids.append(ev.token_id)
    return RunResult(list(records.values()), steps, asdict(system.stats), clock() - t0)


# ------------------------------------------------------------------------------ metrics
def percentiles(values, qs=(50, 99)) -> dict[str, float]:
    """``{"p50": ..., "p99": ..., "mean": ...}`` of ``values`` (linear interpolation)."""
    if len(values) == 0:
        return {f"p{q}": float("nan") for q in qs} | {"mean": float("nan")}
    arr = np.asarray(values, dtype=float)
    return {f"p{q}": float(np.percentile(arr, q)) for q in qs} | {"mean": float(arr.mean())}


def inter_token_gaps(result: RunResult) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every gap between two consecutive tokens of one request: (start, end, length) arrays."""
    a, b = [], []
    for r in result.requests:
        a += r.token_times[:-1]
        b += r.token_times[1:]
    a, b = np.asarray(a), np.asarray(b)
    return a, b, b - a


def prefill_stall_analysis(result: RunResult) -> dict:
    """How much do prefill steps get in the way of decoding?

    Prefill and decode never share a step (see design-notes), so while a prefill step
    runs, every running sequence waits. A gap between two tokens of one request that
    contains a prefill step ("stalled") is longer than one that does not ("clean").
    The linear fit of prefill step time against padded tokens says what one prompt
    token costs, which bounds what chunked prefill could buy.
    """
    a, b, gap = inter_token_gaps(result)
    prefill = [s for s in result.steps if s.is_prefill]
    decode = [s for s in result.steps if not s.is_prefill]
    out: dict = {"num_prefill_steps": len(prefill), "num_decode_steps": len(decode)}
    if len(gap):
        ends = np.sort(np.array([s.t_end for s in prefill]))
        # A prefill step ended inside the gap (a, b]?
        stalled = (np.searchsorted(ends, b, side="right") - np.searchsorted(ends, a, side="right")) > 0
        out["frac_gaps_stalled_by_prefill"] = float(stalled.mean())
        out["itl_ms_clean"] = {k: v * 1e3 for k, v in percentiles(gap[~stalled]).items()}
        out["itl_ms_stalled"] = {k: v * 1e3 for k, v in percentiles(gap[stalled]).items()}
    if decode:
        out["decode_step_ms"] = {k: v * 1e3 for k, v in percentiles([s.t_end - s.t_start for s in decode]).items()}
    if len(prefill) >= 2 and len({s.padded_tokens for s in prefill}) >= 2:
        x = np.array([s.padded_tokens for s in prefill], dtype=float)
        y = np.array([s.t_end - s.t_start for s in prefill]) * 1e3
        slope, intercept = np.polyfit(x, y, 1)
        out["prefill_ms_per_padded_token"] = float(slope)
        out["prefill_ms_fixed"] = float(intercept)
        out["prefill_step_ms"] = percentiles(y)
        out["prefill_padded_tokens"] = percentiles(x)
        out["prefill_max_step_ms"] = float(y.max())
    return out


def summarize(result: RunResult, bytes_per_token: int) -> dict:
    """The brief's metrics for one run (times in ms, memory in MiB)."""
    reqs = result.requests
    finished = [r for r in reqs if r.token_times]
    tokens = sum(len(r.token_times) for r in finished)
    first_arrival = min(r.arrival for r in reqs)
    last_token = max(r.token_times[-1] for r in finished)
    ttft = [(r.token_times[0] - r.arrival) * 1e3 for r in finished]
    latency = [(r.token_times[-1] - r.arrival) * 1e3 for r in finished]
    _, _, gaps = inter_token_gaps(result)
    steps = result.steps
    allocated = sum(s.allocated_tokens for s in steps)
    mib = bytes_per_token / 2**20
    return {
        "requests": len(reqs),
        "output_tokens": tokens,
        "duration_s": last_token - first_arrival,  # first arrival -> last token
        "throughput_tok_s": tokens / (last_token - first_arrival),
        "ttft_ms": percentiles(ttft),
        "itl_ms": {k: v * 1e3 for k, v in percentiles(gaps).items()},
        "latency_ms": percentiles(latency),
        "peak_kv_allocated_mib": max((s.allocated_tokens for s in steps), default=0) * mib,
        "peak_kv_live_mib": max((s.live_tokens for s in steps), default=0) * mib,
        "kv_utilization": sum(s.live_tokens for s in steps) / allocated if allocated else float("nan"),
        "max_concurrent": max((s.num_running for s in steps), default=0),
        "steps": len(steps),
        "stats": result.stats,
        "stalls": prefill_stall_analysis(result),
    }
