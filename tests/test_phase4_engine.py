"""Phase 4: the engine end to end (real GPT-2, paged KV, scheduler).

Every correctness check compares against the Phase 2 contiguous-cache greedy decode
of each request run alone, so a bug anywhere (scheduler, paging, preemption/recompute)
shows up as a different token.
"""

import pytest
import torch

from tokquay.engine import Engine, sample_token
from tokquay.generate import greedy_generate
from tokquay.kv_cache import BlockAllocator
from tokquay.scheduler import RequestRejected, SchedulerConfig
from tokquay.sequence import SamplingParams, SequenceStatus

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_STEPS = 5000
_reference_cache: dict[tuple, list[int]] = {}


def reference(model, prompt: list[int], n: int) -> list[int]:
    key = (tuple(prompt), n)
    if key not in _reference_cache:
        ids = torch.tensor([prompt], device=DEVICE)
        _reference_cache[key] = greedy_generate(model, ids, n, use_cache=True)
    return _reference_cache[key]


def make_engine(model, num_blocks=256, block_size=16, **sched):
    alloc = BlockAllocator(model.cfg, num_blocks, block_size, device=DEVICE)
    return Engine(model, alloc, SchedulerConfig(**sched) if sched else None)


def greedy(n):
    return SamplingParams(max_tokens=n, ignore_eos=True)  # fixed length so it matches the reference


def run(engine, arrivals=(), max_steps=MAX_STEPS):
    """Drive the engine. `arrivals` = [(step, prompt, params)]; returns {seq: (Sequence, first_step, last_step)}."""
    pending = sorted(arrivals, key=lambda a: a[0])
    seqs, first, last = [], {}, {}
    for step in range(max_steps):
        while pending and pending[0][0] <= step:
            _, prompt, params = pending.pop(0)
            seqs.append(engine.add_request(prompt, params))
        if not pending and not engine.has_unfinished():
            break
        for ev in engine.step():
            first.setdefault(ev.seq_id, step)
            last[ev.seq_id] = step
    else:
        pytest.fail(f"engine did not finish within {max_steps} steps")
    return seqs, first, last


def test_single_request_matches_reference(shared_model, text_ids):
    engine = make_engine(shared_model)
    seqs, _, _ = run(engine, [(0, text_ids[:12], greedy(25))])
    assert seqs[0].output_token_ids == reference(shared_model, text_ids[:12], 25)
    assert seqs[0].finish_reason == "length"


def test_request_arriving_mid_generation_is_admitted_without_waiting_for_the_batch(shared_model, text_ids):
    engine = make_engine(shared_model)
    a = engine.add_request(text_ids[:10], greedy(40))
    for _ in range(4):
        engine.step()  # A: prefill + 3 decodes
    assert a.status is SequenceStatus.RUNNING and len(a.output_token_ids) == 4

    b = engine.add_request(text_ids[20:30], greedy(8))  # arrives mid-generation
    events = engine.step()
    assert [e.seq_id for e in events] == [b.seq_id]  # B's prefill runs on the very next step...
    assert a.status is SequenceStatus.RUNNING  # ...while A is still in flight
    events = engine.step()
    assert sorted(e.seq_id for e in events) == [a.seq_id, b.seq_id]  # and they now decode together

    engine.run_until_done()
    assert b.status is SequenceStatus.FINISHED and b.finish_reason == "length"
    assert a.output_token_ids == reference(shared_model, text_ids[:10], 40)
    assert b.output_token_ids == reference(shared_model, text_ids[20:30], 8)
    # B (8 tokens, arrived later) finished before A (40 tokens): requests leave as they finish.
    assert len(engine.scheduler.running) == 0 and engine.allocator.num_free_blocks == engine.allocator.num_blocks


def test_staggered_arrivals_all_match_reference(shared_model, text_ids):
    specs = [  # (arrival step, prompt slice, max_tokens)
        (0, text_ids[0:5], 30),
        (0, text_ids[10:40], 12),
        (3, text_ids[50:58], 25),
        (6, text_ids[70:71], 20),
        (6, text_ids[100:133], 15),
        (11, text_ids[200:217], 18),
    ]
    engine = make_engine(shared_model)
    seqs, first, last = run(engine, [(t, p, greedy(n)) for t, p, n in specs])
    for seq, (_, prompt, n) in zip(seqs, specs):
        assert seq.output_token_ids == reference(shared_model, prompt, n)
    assert engine.stats.max_batch >= 3  # they genuinely overlapped
    # The step-6 arrivals got their first token while the step-0 request was still generating.
    assert first[seqs[3].seq_id] < last[seqs[0].seq_id]


def test_tiny_pool_forces_preemption_but_every_request_finishes_correctly(shared_model, text_ids):
    """16 blocks x 16 tokens = 256 tokens of KV for 8 requests that each want ~70-100."""
    specs = [(0, text_ids[i * 9 : i * 9 + 20 + 5 * i], 40 + 4 * i) for i in range(8)]
    engine = make_engine(shared_model, num_blocks=16, block_size=16)
    seqs, _, _ = run(engine, [(t, p, greedy(n)) for t, p, n in specs])

    assert engine.stats.num_preemptions > 0, "the pool was meant to be too small"
    for seq, (_, prompt, n) in zip(seqs, specs):
        assert seq.status is SequenceStatus.FINISHED
        assert seq.output_token_ids == reference(shared_model, prompt, n)
    assert engine.stats.peak_used_blocks <= 16
    assert engine.allocator.num_free_blocks == 16 and all(c == 0 for c in engine.allocator.ref_counts)


def test_oversized_request_is_rejected_and_does_not_block_the_others(shared_model, text_ids):
    engine = make_engine(shared_model, num_blocks=16, block_size=16)  # 256 tokens of KV
    ok = [engine.add_request(text_ids[i * 8 : i * 8 + 12], greedy(20)) for i in range(3)]
    with pytest.raises(RequestRejected):
        engine.add_request(text_ids[:300], greedy(1))  # needs 19 blocks, the pool has 16
    assert engine.scheduler.num_preemptions == 0 and len(engine.scheduler.waiting) == 3
    engine.run_until_done()  # no deadlock
    for i, seq in enumerate(ok):
        assert seq.output_token_ids == reference(shared_model, text_ids[i * 8 : i * 8 + 12], 20)


def test_request_that_needs_the_entire_pool_still_completes(shared_model, text_ids):
    engine = make_engine(shared_model, num_blocks=16, block_size=16)
    big_prompt = text_ids[:200]
    arrivals = [(0, big_prompt, greedy(57))]  # ctx = 200 + 57 - 1 = 256 = all 16 blocks
    arrivals += [(1, text_ids[300 + 10 * i : 310 + 10 * i], greedy(25)) for i in range(3)]
    seqs, _, _ = run(engine, arrivals)
    assert seqs[0].output_token_ids == reference(shared_model, big_prompt, 57)
    for i, seq in enumerate(seqs[1:]):
        assert seq.output_token_ids == reference(shared_model, text_ids[300 + 10 * i : 310 + 10 * i], 25)
    assert engine.allocator.num_free_blocks == 16


# ------------------------------------------------------------------------------ sampling / stopping
def test_eos_stops_generation_unless_ignored(shared_model, text_ids):
    prompt = text_ids[:10]
    ref = reference(shared_model, prompt, 12)
    eos = ref[3]
    assert eos not in ref[:3], "pick a prompt whose 4th token is new"
    alloc = BlockAllocator(shared_model.cfg, 64, 16, device=DEVICE)
    engine = Engine(shared_model, alloc, eos_token_id=eos)

    stopped = engine.add_request(prompt, SamplingParams(max_tokens=12))
    ignoring = engine.add_request(prompt, SamplingParams(max_tokens=12, ignore_eos=True))
    engine.run_until_done()
    assert stopped.output_token_ids == ref[:4] and stopped.finish_reason == "stop"  # EOS token is included
    assert ignoring.output_token_ids == ref and ignoring.finish_reason == "length"


def test_seeded_sampling_is_reproducible_and_mixes_with_greedy_rows(shared_model, text_ids):
    prompt = text_ids[:10]
    params = SamplingParams(max_tokens=20, temperature=0.9, top_k=40, seed=1234, ignore_eos=True)

    def go(seed_params):
        engine = make_engine(shared_model)
        s = engine.add_request(prompt, seed_params)
        g = engine.add_request(prompt, greedy(20))  # a greedy row batched with the sampled one
        engine.run_until_done()
        return s.output_token_ids, g.output_token_ids

    s1, g1 = go(params)
    s2, g2 = go(params)
    s3, _ = go(SamplingParams(max_tokens=20, temperature=0.9, top_k=40, seed=99, ignore_eos=True))
    assert s1 == s2 and g1 == g2 == reference(shared_model, prompt, 20)  # reproducible; greedy unaffected
    assert s1 != s3 and s1 != g1  # different seed / not greedy -> different text


def test_top_k_1_equals_greedy(shared_model, text_ids):
    engine = make_engine(shared_model)
    s = engine.add_request(text_ids[:10], SamplingParams(max_tokens=15, temperature=1.5, top_k=1, ignore_eos=True))
    engine.run_until_done()
    assert s.output_token_ids == reference(shared_model, text_ids[:10], 15)


def test_sample_token_respects_top_k_and_temperature():
    logits = torch.tensor([5.0, 4.0, 3.0, 0.0, -1.0, -2.0, -3.0, -4.0])
    gen = torch.Generator().manual_seed(0)
    draws = {sample_token(logits, SamplingParams(temperature=2.0, top_k=3), gen) for _ in range(300)}
    assert draws == {0, 1, 2}  # never outside the top 3, and all of them get picked
    cold = {sample_token(logits, SamplingParams(temperature=0.01), gen) for _ in range(50)}
    assert cold == {0}  # near-zero temperature is effectively greedy
    assert sample_token(logits, SamplingParams(), None) == 0  # temperature 0 needs no generator
    hot = {sample_token(logits, SamplingParams(temperature=50.0), gen) for _ in range(400)}
    assert len(hot) >= 6  # very high temperature is close to uniform


def test_sampling_params_validation():
    for bad in ({"max_tokens": 0}, {"temperature": -1.0}, {"top_k": -1}):
        with pytest.raises(ValueError):
            SamplingParams(**bad)


def test_stats_are_recorded(shared_model, text_ids):
    engine = make_engine(shared_model)
    for i in range(3):
        engine.add_request(text_ids[i * 7 : i * 7 + 9], greedy(6))
    engine.run_until_done()
    st = engine.stats
    assert st.tokens_generated == 18 and st.max_batch == 3
    assert st.prefill_steps == 1 and st.decode_steps == 5 and st.steps == 6
    assert 0 < st.peak_used_blocks <= 3
    assert engine.step() == []  # idle engine is a no-op
