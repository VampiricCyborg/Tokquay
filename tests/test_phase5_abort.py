"""Phase 5: cancelling a request (its client disconnected) on the synchronous engine.

An aborted request must give every KV block back and must not disturb the others:
the survivors still have to match the single-request reference token for token.
"""

import random

import pytest
import torch

from tokquay.engine import Engine
from tokquay.kv_cache import BlockAllocator
from tokquay.scheduler import SchedulerConfig
from tokquay.sequence import SamplingParams, SequenceStatus

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_engine(model, num_blocks=256, block_size=16):
    return Engine(model, BlockAllocator(model.cfg, num_blocks, block_size, device=DEVICE))


def greedy(n):
    return SamplingParams(max_tokens=n, ignore_eos=True)


def assert_pool_clean(engine):
    a = engine.allocator
    assert a.num_free_blocks == a.num_blocks and all(c == 0 for c in a.ref_counts)
    assert not engine.scheduler.running and not engine.scheduler.waiting and not engine._rngs


def test_abort_a_queued_request(shared_model, text_ids, greedy_reference):
    engine = make_engine(shared_model)
    a = engine.add_request(text_ids[:10], greedy(20))
    b = engine.add_request(text_ids[20:30], greedy(20))
    engine.abort_request(b)  # still waiting: it never ran
    assert b.status is SequenceStatus.FINISHED and b.finish_reason == "abort" and b.output_token_ids == []
    assert list(engine.scheduler.waiting) == [a]
    engine.run_until_done()
    assert a.output_token_ids == greedy_reference(text_ids[:10], 20)
    assert_pool_clean(engine)


def test_abort_a_running_request_frees_its_blocks_and_spares_the_others(shared_model, text_ids, greedy_reference):
    engine = make_engine(shared_model)
    specs = [(text_ids[0:10], 30), (text_ids[20:45], 30), (text_ids[60:70], 30)]
    a, b, c = (engine.add_request(p, greedy(n)) for p, n in specs)
    for _ in range(6):
        engine.step()
    assert b.status is SequenceStatus.RUNNING and b.block_table
    used = engine.allocator.num_used_blocks
    held = len(b.block_table)

    engine.abort_request(b)
    assert b.status is SequenceStatus.FINISHED and b.finish_reason == "abort" and b.block_table == []
    assert engine.allocator.num_used_blocks == used - held
    assert b not in engine.scheduler.running

    engine.run_until_done()
    for seq, (prompt, n) in zip((a, c), (specs[0], specs[2])):
        assert seq.output_token_ids == greedy_reference(prompt, n)
    assert b.output_token_ids == greedy_reference(*specs[1])[: len(b.output_token_ids)]  # what it did emit was right
    assert_pool_clean(engine)


def test_abort_a_preempted_request_that_is_waiting_to_be_recomputed(shared_model, text_ids, greedy_reference):
    engine = make_engine(shared_model, num_blocks=12)
    specs = [(text_ids[i * 9 : i * 9 + 20 + 5 * i], 30 + 4 * i) for i in range(6)]
    seqs = [engine.add_request(p, greedy(n)) for p, n in specs]

    victim = None
    for _ in range(2000):
        engine.step()
        victim = next((s for s in engine.scheduler.waiting if s.output_token_ids), None)
        if victim is not None:
            break
    assert victim is not None, "the pool was meant to be too small"
    assert victim.block_table == []  # preempted: it owns nothing while it waits

    engine.abort_request(victim)
    assert victim.status is SequenceStatus.FINISHED and victim not in engine.scheduler.waiting
    engine.run_until_done()
    for seq, (prompt, n) in zip(seqs, specs):
        if seq is not victim:
            assert seq.output_token_ids == greedy_reference(prompt, n)
    assert_pool_clean(engine)


def test_abort_is_idempotent_and_ignores_finished_requests(shared_model, text_ids, greedy_reference):
    engine = make_engine(shared_model)
    a = engine.add_request(text_ids[:10], greedy(5))
    engine.run_until_done()
    assert a.finish_reason == "length"
    engine.abort_request(a)  # already finished: nothing changes
    assert a.finish_reason == "length" and a.output_token_ids == greedy_reference(text_ids[:10], 5)

    b = engine.add_request(text_ids[20:30], greedy(5))
    engine.abort_request(b)
    engine.abort_request(b)  # twice is fine
    assert b.finish_reason == "abort"
    assert_pool_clean(engine)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_aborts_never_corrupt_survivors_or_leak_blocks(shared_model, text_ids, greedy_reference, seed):
    rng = random.Random(seed)
    engine = make_engine(shared_model, num_blocks=10)  # small enough that preemption and aborts interleave
    specs = [(text_ids[i * 11 : i * 11 + rng.randint(5, 30)], rng.randint(8, 30)) for i in range(8)]
    seqs = [engine.add_request(p, greedy(n)) for p, n in specs]

    # Every request needs at least 8 steps, so all of them are still alive in steps 1-6:
    # three aborts are guaranteed there. Later steps get occasional random ones as well.
    guaranteed = set(rng.sample(range(1, 7), 3))
    for step in range(5000):
        if not engine.has_unfinished():
            break
        engine.step()
        live = [s for s in seqs if s.status is not SequenceStatus.FINISHED]
        if live and (step in guaranteed or (step > 6 and rng.random() < 0.05)):
            engine.abort_request(rng.choice(live))
    else:
        pytest.fail("engine did not finish")

    for seq, (prompt, n) in zip(seqs, specs):
        ref = greedy_reference(prompt, n)
        if seq.finish_reason == "abort":
            assert seq.output_token_ids == ref[: len(seq.output_token_ids)]
        else:
            assert seq.finish_reason == "length" and seq.output_token_ids == ref
    assert sum(s.finish_reason == "abort" for s in seqs) >= 3
    assert_pool_clean(engine)
