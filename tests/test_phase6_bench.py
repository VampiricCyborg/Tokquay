"""Phase 6: the benchmark measures what it claims to.

The numbers in the README come from ``bench/``, so the workload generator and every
metric are checked here: the metric code against a scripted server on a virtual clock
(where each number can be worked out by hand), the workload against its statistics,
and both real systems end to end on a small trace.
"""

import math
import types

import numpy as np
import pytest
import torch

from bench.harness import (
    RequestRecord,
    RunResult,
    StepRecord,
    inter_token_gaps,
    percentiles,
    prefill_stall_analysis,
    run_trace,
    summarize,
)
from bench.workload import VOCAB_SIZE, RequestSpec, make_workload
from tokquay.baseline import StaticBatchEngine
from tokquay.engine import Engine, EngineStats, KVSnapshot, StepInfo, TokenEvent
from tokquay.kv_cache import BlockAllocator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------------------ workload
def test_workload_is_deterministic_and_within_the_briefs_ranges():
    a, b = make_workload(2.0, 300, seed=5), make_workload(2.0, 300, seed=5)
    assert a == b
    assert make_workload(2.0, 300, seed=6) != a
    assert all(16 <= len(r.prompt_ids) <= 256 and 16 <= r.output_len <= 256 for r in a)
    assert all(0 <= t < VOCAB_SIZE - 1 for r in a for t in r.prompt_ids)  # never <|endoftext|>
    assert [r.arrival for r in a] == sorted(r.arrival for r in a)
    lens = [len(r.prompt_ids) for r in a]
    assert 16 <= min(lens) < 40 and max(lens) > 230  # spread over the whole range, not clumped


def test_workloads_at_different_rates_contain_the_same_requests_compressed_in_time():
    slow, fast = make_workload(1.0, 200, seed=3), make_workload(4.0, 200, seed=3)
    assert [(r.prompt_ids, r.output_len) for r in slow] == [(r.prompt_ids, r.output_len) for r in fast]
    for s, f in zip(slow, fast):
        assert s.arrival == pytest.approx(4 * f.arrival)


def test_arrivals_form_a_poisson_process_at_the_requested_rate():
    rate = 2.5
    w = make_workload(rate, 6000, seed=1)
    gaps = np.diff([0.0] + [r.arrival for r in w])
    assert gaps.mean() == pytest.approx(1 / rate, rel=0.05)
    assert gaps.std() == pytest.approx(1 / rate, rel=0.08)  # exponential gaps: std equals the mean
    assert (make_workload(math.inf, 5, seed=1)[-1].arrival) == 0.0  # inf = everything at once
    for bad in ({"rate": 0, "n": 5}, {"rate": 1, "n": 0}):
        with pytest.raises(ValueError):
            make_workload(**bad)


# ------------------------------------------------------------------------------ metrics, by hand
class FakeClock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, dt):
        self.t += dt


class FakeSystem:
    """A scripted server on a virtual clock: 2 concurrent requests, FCFS. A step that admits
    waiting requests is a prefill step (0.5 s); any other is a decode step (0.1 s). Every request
    in a step gets one token."""

    def __init__(self, clock):
        self.clock, self.slots = clock, 2
        self.waiting, self.running = [], []
        self.stats, self.last_step, self._n = EngineStats(), None, 0

    def add_request(self, prompt_ids, params):
        seq = types.SimpleNamespace(seq_id=self._n, left=params.max_tokens)
        self._n += 1
        self.waiting.append(seq)
        return seq

    def has_unfinished(self):
        return bool(self.waiting or self.running)

    def step(self):
        admit = []
        while self.waiting and len(self.running) + len(admit) < self.slots:
            admit.append(self.waiting.pop(0))
        self.running += admit
        batch, prefill = (admit, True) if admit else (list(self.running), False)
        self.clock.t += 0.5 if prefill else 0.1
        events = []
        for s in batch:
            s.left -= 1
            events.append(TokenEvent(s.seq_id, 7, "length" if s.left == 0 else None))
        kv = KVSnapshot(len(self.running) * 100, len(self.running) * 10, len(self.running))
        self.last_step = StepInfo(prefill, len(batch), 3 * len(batch) if prefill else len(batch), kv)
        self.running = [s for s in self.running if s.left > 0]
        return events


def hand_worked_scenario():
    clock = FakeClock()
    workload = [
        RequestSpec(0.0, (1,), 3),
        RequestSpec(0.25, (1,), 2),  # arrives during the first request's prefill
        RequestSpec(5.0, (1,), 1),  # arrives after everything drained: the driver must idle until then
    ]
    return run_trace(FakeSystem(clock), workload, clock=clock.now, sleep=clock.sleep)


def test_run_trace_timestamps_tokens_when_their_step_ends():
    result = hand_worked_scenario()
    # t=0.0 r0 prefill -> 0.5 | r1 (arrived 0.25) prefill -> 1.0 | decode both -> 1.1 (r1 done) | decode r0 -> 1.2
    # then idle until 5.0 | r2 prefill -> 5.5
    r0, r1, r2 = result.requests
    assert r0.token_times == pytest.approx([0.5, 1.1, 1.2])
    assert r1.token_times == pytest.approx([1.0, 1.1])
    assert r2.token_times == pytest.approx([5.5])
    assert [(s.is_prefill, round(s.t_end - s.t_start, 3)) for s in result.steps] == [
        (True, 0.5), (True, 0.5), (False, 0.1), (False, 0.1), (True, 0.5),
    ]  # fmt: skip


def test_summary_metrics_match_a_hand_calculation():
    m = summarize(hand_worked_scenario(), bytes_per_token=2**20)  # 1 MiB per token slot
    # TTFT is measured from the *scheduled* arrival: r1 arrived at 0.25 but was only picked up
    # after r0's 0.5 s prefill, and that wait is charged to it: 1.0 - 0.25 = 0.75 s.
    assert m["ttft_ms"]["p50"] == pytest.approx(500)
    assert m["ttft_ms"]["mean"] == pytest.approx((500 + 750 + 500) / 3)
    assert m["ttft_ms"]["p99"] == pytest.approx(745)  # linear interpolation between 500 and 750
    # Gaps between consecutive tokens of one request: r0: 0.6, 0.1; r1: 0.1.
    assert m["itl_ms"]["p50"] == pytest.approx(100)
    assert m["itl_ms"]["mean"] == pytest.approx((600 + 100 + 100) / 3)
    # 6 tokens from the first arrival (0.0) to the last token (5.5).
    assert m["output_tokens"] == 6 and m["duration_s"] == pytest.approx(5.5)
    assert m["throughput_tok_s"] == pytest.approx(6 / 5.5)
    assert m["latency_ms"]["p50"] == pytest.approx(850)  # last token minus arrival: r0 1200, r1 850, r2 500 ms
    assert m["max_concurrent"] == 2 and m["steps"] == 5
    assert m["peak_kv_allocated_mib"] == 200 and m["peak_kv_live_mib"] == 20
    assert m["kv_utilization"] == pytest.approx(70 / 700)  # sum(live) / sum(allocated) over the 5 steps


def test_prefill_stall_analysis_separates_gaps_that_contain_a_prefill_step():
    stalls = summarize(hand_worked_scenario(), 1)["stalls"]
    # Only r0's first gap (0.5, 1.1] contains a prefill step (r1's, ending at 1.0).
    assert stalls["frac_gaps_stalled_by_prefill"] == pytest.approx(1 / 3)
    assert stalls["itl_ms_stalled"]["p50"] == pytest.approx(600)
    assert stalls["itl_ms_clean"]["p50"] == pytest.approx(100)
    assert stalls["num_prefill_steps"] == 3 and stalls["num_decode_steps"] == 2
    assert stalls["decode_step_ms"]["mean"] == pytest.approx(100)
    assert "prefill_ms_per_padded_token" not in stalls  # all prefills had 3 padded tokens: nothing to fit


def test_prefill_cost_per_token_is_recovered_from_the_step_times():
    # Prefill steps that cost 20 ms + 0.05 ms per padded token, exactly.
    recs, t = [], 0.0
    for tokens in (100, 400, 900, 2000):
        dur = (20 + 0.05 * tokens) / 1e3
        recs.append(StepRecord(t, t + dur, True, 1, tokens, 0, 0, 1))
        t += dur
    result = RunResult([RequestRecord(0.0, 1, 2, [0.1, 0.2])], recs, {}, t)
    out = prefill_stall_analysis(result)
    assert out["prefill_ms_per_padded_token"] == pytest.approx(0.05)
    assert out["prefill_ms_fixed"] == pytest.approx(20)
    assert out["prefill_max_step_ms"] == pytest.approx(120)


def test_percentiles_use_linear_interpolation_and_handle_empty_input():
    p = percentiles(list(range(1, 101)))
    assert p == {"p50": pytest.approx(50.5), "p99": pytest.approx(99.01), "mean": pytest.approx(50.5)}
    assert all(math.isnan(v) for v in percentiles([]).values())
    a, b, gap = inter_token_gaps(hand_worked_scenario())
    assert list(gap.round(3)) == [0.6, 0.1, 0.1]


def test_a_stuck_system_raises_instead_of_spinning_forever():
    class Stuck:
        stats, last_step = EngineStats(), None

        def add_request(self, ids, params):
            return types.SimpleNamespace(seq_id=0)

        def has_unfinished(self):
            return True

        def step(self):
            return []

    with pytest.raises(RuntimeError, match="deadlock"):
        run_trace(Stuck(), [RequestSpec(0.0, (1,), 1)])


# ------------------------------------------------------------------------------ real systems, small trace
def test_both_systems_serve_a_small_trace_correctly_and_paging_uses_less_kv_memory(
    shared_model, greedy_reference
):
    workload = make_workload(math.inf, 6, seed=2, prompt_len=(5, 20), output_len=(4, 10))
    tok_engine = Engine(shared_model, BlockAllocator(shared_model.cfg, 64, 16, device=DEVICE))
    static = StaticBatchEngine(shared_model, batch_size=4, max_len=64, device=DEVICE)
    results = {"tokquay": run_trace(tok_engine, workload), "static": run_trace(static, workload)}
    bpt = {"tokquay": tok_engine.allocator.bytes_per_block // 16, "static": static.bytes_per_token}

    for name, result in results.items():
        assert len(result.requests) == 6
        for rec, spec in zip(result.requests, workload):
            assert len(rec.token_times) == spec.output_len  # exactly the requested length
            assert rec.token_times == sorted(rec.token_times) and rec.token_times[0] >= rec.arrival
            assert rec.token_ids == greedy_reference(list(spec.prompt_ids), spec.output_len)  # and the right tokens
    m = {name: summarize(r, bpt[name]) for name, r in results.items()}
    assert m["tokquay"]["max_concurrent"] == 6 and m["static"]["max_concurrent"] == 4  # the static batch is capped
    # Same requests, but the static cache reserves max_len per row and paging does not.
    assert m["tokquay"]["peak_kv_allocated_mib"] < m["static"]["peak_kv_allocated_mib"]
    assert m["tokquay"]["kv_utilization"] > m["static"]["kv_utilization"]
    assert m["tokquay"]["throughput_tok_s"] > 0 and m["static"]["throughput_tok_s"] > 0
