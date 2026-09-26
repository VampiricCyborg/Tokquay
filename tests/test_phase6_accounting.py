"""Phase 6: the work and memory accounting the benchmark reports is exact.

The benchmark quotes "how much of the forward-pass work was recomputation" and KV
memory numbers, so they are checked against closed-form expectations, not just
sanity-checked.
"""

import torch

from tokquay.engine import Engine, KVSnapshot
from tokquay.kv_cache import BlockAllocator
from tokquay.scheduler import SchedulerConfig
from tokquay.sequence import SamplingParams

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def greedy(n):
    return SamplingParams(max_tokens=n, ignore_eos=True)


def make_engine(model, num_blocks, block_size=16):
    return Engine(model, BlockAllocator(model.cfg, num_blocks, block_size, device=DEVICE))


def needed_positions(specs):
    """Positions that must be computed at minimum: each prompt, plus every generated
    token except the last (the last is returned to the caller and never fed back)."""
    return sum(len(p) + n - 1 for p, n in specs)


def test_positions_computed_is_exactly_the_necessary_work_when_nothing_is_preempted(shared_model, text_ids):
    specs = [(text_ids[i * 9 : i * 9 + 8 + 3 * i], 6 + 2 * i) for i in range(5)]
    engine = make_engine(shared_model, num_blocks=64)
    for p, n in specs:
        engine.add_request(p, greedy(n))
    engine.run_until_done()
    st = engine.stats
    assert st.num_preemptions == 0 and st.positions_discarded == 0
    assert st.positions_computed == needed_positions(specs)
    assert st.positions_padded >= st.positions_computed  # padding can only add work


def test_preemption_adds_exactly_the_discarded_positions(shared_model, text_ids):
    """Every K/V position a preemption throws away is computed a second time, no more, no less."""
    specs = [(text_ids[i * 9 : i * 9 + 18 + 3 * i], 24 + 2 * i) for i in range(5)]
    engine = make_engine(shared_model, num_blocks=8)  # 128 tokens for 5 requests wanting ~45 each
    for p, n in specs:
        engine.add_request(p, greedy(n))
    engine.run_until_done()
    st = engine.stats
    assert st.num_preemptions > 0 and st.positions_discarded > 0, "the pool was meant to be too small"
    assert st.positions_computed == needed_positions(specs) + st.positions_discarded
    assert st.positions_discarded == engine.scheduler.num_discarded_positions


def test_last_step_and_kv_snapshot_describe_the_step_that_just_ran(shared_model, text_ids):
    engine = make_engine(shared_model, num_blocks=64, block_size=16)
    assert engine.last_step is None and engine.kv_snapshot().num_seqs == 0
    engine.add_request(text_ids[:5], greedy(4))
    engine.add_request(text_ids[10:30], greedy(4))

    engine.step()  # prefill of both
    assert (engine.last_step.is_prefill, engine.last_step.num_seqs) == (True, 2)
    assert engine.last_step.padded_tokens == 2 * 20  # batch x longest prompt
    snap = engine.last_step.kv
    assert snap == engine.kv_snapshot()  # nothing finished, so the step's peak is also "now"
    assert snap.num_seqs == 2 and snap.live_tokens == 5 + 20
    assert snap.allocated_tokens == (1 + 2) * 16  # 1 block for the 5-token prompt, 2 for the 20-token one
    assert snap.live_tokens <= snap.allocated_tokens

    engine.step()  # decode of both
    assert (engine.last_step.is_prefill, engine.last_step.num_seqs, engine.last_step.padded_tokens) == (False, 2, 2)
    engine.run_until_done()
    assert engine.kv_snapshot() == KVSnapshot(0, 0, 0)  # everything freed
    engine.step()
    assert engine.last_step is None  # idle


def test_step_snapshot_is_taken_at_the_peak_before_finished_sequences_free_their_blocks(shared_model, text_ids):
    engine = make_engine(shared_model, num_blocks=64, block_size=16)
    engine.add_request(text_ids[:5], greedy(2))  # finishes on the first decode step
    engine.add_request(text_ids[10:30], greedy(6))
    engine.step()  # prefill
    engine.step()  # decode: the first request emits its 2nd token and is done
    at_peak, after = engine.last_step.kv, engine.kv_snapshot()
    assert at_peak.num_seqs == 2 and after.num_seqs == 1  # a snapshot taken afterwards would miss it
    assert at_peak.allocated_tokens == after.allocated_tokens + 16  # its one block
    assert at_peak.live_tokens == (5 + 1) + (20 + 1)  # both had K/V written this step
