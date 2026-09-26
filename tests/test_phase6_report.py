"""Phase 6: the report turns saved results into tables and plots without inventing anything.

Run on synthetic results (no GPU), so every number in the output can be traced to an input.
"""

import types

import pytest

from bench.report import THEMES, build_markdown, headline_table, report


def summary(tp, ttft=(500, 900), itl=(50, 120), kv=400, live=200, util=0.5, conc=8, rate_mult=1.0, **extra):
    s = {
        "requests": 200, "output_tokens": 26000, "duration_s": 100.0, "throughput_tok_s": tp,
        "ttft_ms": {"p50": ttft[0], "p99": ttft[1], "mean": 0}, "itl_ms": {"p50": itl[0], "p99": itl[1], "mean": 0},
        "peak_kv_allocated_mib": kv, "peak_kv_live_mib": live, "kv_utilization": util, "max_concurrent": conc,
        "stats": {"num_preemptions": 0, "positions_computed": 1000, "positions_discarded": 0},
        "stalls": {"frac_gaps_stalled_by_prefill": 0.25, "itl_ms_clean": {"p50": 50, "p99": 60},
                   "itl_ms_stalled": {"p50": 200, "p99": 400}, "num_prefill_steps": 100, "num_decode_steps": 900,
                   "prefill_step_ms": {"p50": 30, "p99": 90, "mean": 40}, "prefill_max_step_ms": 350.0,
                   "prefill_padded_tokens": {"p50": 150, "p99": 800, "mean": 200},
                   "decode_step_ms": {"p50": 40, "p99": 70, "mean": 45}},
        "config": {"kv_reserved_mib": 1152.0},
        "gpu": {"peak_allocated_mib": 1700.0, "peak_reserved_mib": 2048.0, "total_mib": 4096.0, "alloc_retries": 0},
    }  # fmt: skip
    s.update(extra)
    return s


def fake_store(full=True):
    runs = {}
    for rate, (t_tp, s_tp) in {1.0: (130.0, 128.0), 2.0: (255.0, 200.0), 4.0: (380.0, 210.0)}.items():
        runs[f"sweep/tokquay/rate={rate:g}"] = {"rate": rate, "summary": summary(t_tp, conc=40)}
        runs[f"sweep/static/rate={rate:g}"] = {"rate": rate, "summary": summary(s_tp, ttft=(5000, 30000), kv=1152, conc=32)}
    if full:
        runs["calibrate/tokquay"] = {"summary": summary(384.5)}
        runs["calibrate/static"] = {"summary": summary(223.4)}
        for bs in (16, 32):
            runs[f"blocksize/bs={bs}/rate=2/tokens=16384"] = {"rate": 2.0, "summary": summary(250.0 + bs)}
        for tokens, mib in ((16384, 1152), (8192, 576), (4096, 288)):
            for system in ("tokquay", "static"):
                stats = {"num_preemptions": 7, "positions_computed": 2000, "positions_discarded": 100} if system == "tokquay" else {}
                runs[f"pool/{system}/tokens={tokens}"] = {
                    "rate": 4.0,
                    "summary": summary(300 if system == "tokquay" else 150, config={"kv_reserved_mib": float(mib)}, stats=stats),
                }
        runs["sweep/tokquay/rate=2#repeat"] = {"rate": 2.0, "summary": summary(251.0)}
    meta = {"gpu": "Test GPU", "torch": "0.0", "dtype": "float32", "date": "2026-01-01", "memory_fraction": 0.8,
            "cpu_affinity": "0xff", "high_priority": True}
    return types.SimpleNamespace(data={"meta": meta, "runs": runs})


def test_markdown_carries_every_number_of_the_plots_and_the_briefs_metrics():
    md = build_markdown(fake_store())
    assert "| 2 | 200.0 | 255.0 | 1.27x |" in md  # the throughput table behind the main plot, with the ratio
    assert "### At 4 requests/s" in md
    assert "Throughput (tok/s) | 210.0 | 380.0" in md  # brief's headline table: static then Tokquay
    assert "TTFT p50 / p99 (ms) | 5,000 / 30,000 | 500 / 900" in md
    assert "Max concurrent seqs | 32 | 40" in md
    assert "Saturated throughput" in md and "384.5" in md
    assert "Block size, 16 vs 32" in md and "Shrinking the KV budget" in md
    assert "5.0%" in md  # 100 discarded of 2000 computed: the recompute share, from the raw counters
    assert "Run time in prefill steps" in md
    assert "Token gaps containing a prefill" in md and "all at t=0" in md  # the stall table, with the calibration row
    assert "process pinned to CPUs 0xff at high priority" in md
    assert "Repeat runs" in md and "251.0" in md
    assert "GPU memory health" in md and "| sweep/static/rate=2 | 1700 | 2048 | 50% | 0 |" in md
    assert "capped at 80%" in md


def test_a_partial_result_file_still_produces_a_report(tmp_path):
    report(fake_store(full=False), tmp_path)  # only the sweep: no calibration, block size or pool sections
    md = (tmp_path / "results.md").read_text(encoding="utf-8")
    assert "Arrival-rate sweep" in md and "Block size" not in md and "Shrinking" not in md
    assert not list((tmp_path / "img").glob("throughput_vs_kv_budget*"))


@pytest.mark.parametrize("theme", list(THEMES))
def test_plots_are_written_for_both_themes(tmp_path, theme):
    report(fake_store(), tmp_path)
    for name in ("throughput_vs_rate", "latency_vs_rate", "throughput_vs_kv_budget"):
        f = tmp_path / "img" / f"{name}_{theme}.png"
        assert f.exists() and f.stat().st_size > 5_000


def test_headline_table_orders_columns_static_then_tokquay():
    store = fake_store()
    table = headline_table(store.data["runs"]["sweep/static/rate=2"]["summary"], store.data["runs"]["sweep/tokquay/rate=2"]["summary"])
    assert table.splitlines()[0] == "| Metric | Static baseline | Tokquay |"
