"""Phase 4: scheduler policy, tested without a model.

`fake_step` stands in for `Engine.step`: it takes the scheduler's batch, pretends
the forward consumed every uncached token, and appends a dummy token. That is all
the scheduler can observe, so its behaviour (admission, preemption, budgets,
rejection, termination) can be tested exactly and fast.
"""

import pytest
from hypothesis import given, settings, strategies as st

from tokquay.kv_cache import BlockAllocator, cdiv
from tokquay.model import GPT2Config
from tokquay.scheduler import RequestRejected, Scheduler, SchedulerConfig
from tokquay.sequence import SamplingParams, Sequence, SequenceStatus

pytestmark = pytest.mark.timeout(60)  # these are pure logic: a hang means a scheduler livelock

TINY = GPT2Config(vocab_size=8, n_positions=64, n_embd=4, n_layer=1, n_head=1)


def make(num_blocks=64, block_size=4, **cfg):
    cfg.setdefault("max_model_len", 10_000)
    cfg.setdefault("max_num_batched_tokens", 10_000)
    cfg.setdefault("watermark_blocks", 0)
    alloc = BlockAllocator(TINY, num_blocks, block_size, device="cpu")
    return Scheduler(alloc, SchedulerConfig(**cfg))


def req(sched, seq_id, prompt_len, max_tokens):
    seq = Sequence(seq_id, [1] * prompt_len, sampling=SamplingParams(max_tokens=max_tokens))
    sched.add_request(seq)
    return seq


def fake_step(sched):
    batch = sched.schedule()
    if batch is None:
        return None
    for s in batch.seqs:
        s.num_cached_tokens = s.num_tokens  # the forward wrote K/V for every uncached token
        s.append_token(0)
        if len(s.output_token_ids) >= s.sampling.max_tokens:
            sched.finish(s, "length")
    return batch


def ids(seqs):
    return [s.seq_id for s in seqs]


# ------------------------------------------------------------------- continuous batching
def test_fcfs_admission_and_prefill_priority():
    sched = make()
    a, b, c = (req(sched, i, 5, 10) for i in range(3))
    batch = fake_step(sched)
    assert batch.is_prefill and ids(batch.seqs) == [0, 1, 2]  # arrival order, one prefill step
    assert all(s.status is SequenceStatus.RUNNING for s in (a, b, c))
    assert not fake_step(sched).is_prefill  # nothing waiting -> decode


def test_request_arriving_mid_generation_is_admitted_without_draining():
    sched = make()
    a = req(sched, 0, 5, 30)
    for _ in range(4):
        fake_step(sched)  # A prefills, then decodes
    assert a.status is SequenceStatus.RUNNING and len(a.output_token_ids) == 4

    b = req(sched, 1, 5, 30)  # arrives while A is mid-generation
    batch = fake_step(sched)
    assert batch.is_prefill and ids(batch.seqs) == [1]  # very next step, A not drained
    assert a.status is SequenceStatus.RUNNING
    batch = fake_step(sched)
    assert not batch.is_prefill and ids(batch.seqs) == [0, 1]  # both now decode together


def test_finished_requests_leave_immediately_and_free_their_blocks():
    sched = make(num_blocks=8)
    short = req(sched, 0, 4, 2)
    long = req(sched, 1, 4, 20)
    fake_step(sched)  # prefill both
    fake_step(sched)  # short emits its 2nd token and finishes
    assert short.status is SequenceStatus.FINISHED and short.block_table == []
    assert ids(sched.running) == [1]
    assert ids(fake_step(sched).seqs) == [1]  # the batch shrank; nobody waited for `long`
    assert sched.allocator.num_used_blocks == len(long.block_table)


def test_max_num_seqs_limits_admission_and_slots_are_reused_in_order():
    sched = make(max_num_seqs=2)
    seqs = [req(sched, i, 4, 3) for i in range(5)]
    assert ids(fake_step(sched).seqs) == [0, 1]
    admitted_later = []
    while sched.has_unfinished():
        batch = fake_step(sched)
        assert len(sched.running) <= 2
        if batch.is_prefill:
            admitted_later += ids(batch.seqs)
    assert admitted_later == [2, 3, 4]  # FCFS
    assert all(s.status is SequenceStatus.FINISHED for s in seqs)


def test_token_budget_counts_padded_tokens_and_never_skips_the_queue_head():
    sched = make(max_num_batched_tokens=64)
    req(sched, 0, 30, 2)  # 1 x 30 = 30
    req(sched, 1, 30, 2)  # 2 x 30 = 60 <= 64
    req(sched, 2, 30, 2)  # 3 x 30 = 90 > 64
    assert ids(fake_step(sched).seqs) == [0, 1]

    sched = make(max_num_batched_tokens=64)
    req(sched, 0, 30, 2)
    req(sched, 1, 60, 2)  # padded 2 x 60 = 120 > 64: blocks the queue
    req(sched, 2, 5, 2)  # would fit, but must not jump ahead of request 1
    batch = fake_step(sched)
    assert ids(batch.seqs) == [0]
    assert batch.padded_tokens <= 64


def test_a_head_of_queue_that_lacks_blocks_is_not_overtaken_by_a_smaller_request():
    sched = make(num_blocks=8, block_size=4)
    a = req(sched, 0, 12, 20)  # 3 blocks
    fake_step(sched)  # A is running; 5 blocks free
    big = req(sched, 1, 24, 2)  # needs 6 blocks: does not fit yet
    small = req(sched, 2, 4, 2)  # needs 1 block: would fit, but arrived later
    batch = sched.schedule()
    assert not batch.is_prefill and ids(batch.seqs) == [0]  # decode; nothing was admitted
    assert small.status is SequenceStatus.WAITING and big.status is SequenceStatus.WAITING
    assert ids(sched.waiting) == [1, 2]  # order untouched


# ------------------------------------------------------------------- preemption
def preemption_scenario():
    """6 blocks of 4 tokens. A(8 tokens) and B(8) take 2 blocks each, C(4) takes 1."""
    sched = make(num_blocks=6, block_size=4)
    a, b, c = req(sched, 0, 8, 16), req(sched, 1, 8, 16), req(sched, 2, 4, 16)
    prefill = fake_step(sched)
    assert prefill.is_prefill and sched.allocator.num_free_blocks == 1
    return sched, a, b, c


def test_preemption_evicts_the_newest_and_requeues_it_at_the_front():
    sched, a, b, c = preemption_scenario()
    d = req(sched, 3, 10, 2)  # another waiting request, arrived after C
    batch = sched.schedule()  # decode: A takes the last block, B has none left -> evict newest (C)

    assert ids(batch.seqs) == [0, 1] and ids(batch.preempted) == [2]
    assert c.status is SequenceStatus.WAITING and c.block_table == [] and c.num_cached_tokens == 0
    assert c.output_token_ids == [0]  # progress is kept; only the cache is recomputed
    assert ids(sched.waiting) == [2, 3]  # C is ahead of the later arrival D
    assert sched.num_preemptions == 1
    assert len(a.block_table) == 3 and len(b.block_table) == 3 and sched.allocator.num_free_blocks == 0


def test_preempted_sequence_is_recomputed_from_prompt_plus_output_and_finishes():
    sched, a, b, c = preemption_scenario()
    sched.schedule()  # preempts C
    recomputed = None
    while sched.has_unfinished():
        batch = fake_step(sched)
        if batch.is_prefill and 2 in ids(batch.seqs):
            recomputed = batch
    assert recomputed is not None
    assert all(s.status is SequenceStatus.FINISHED for s in (a, b, c))
    assert all(len(s.output_token_ids) == 16 for s in (a, b, c))
    assert sched.allocator.num_free_blocks == sched.allocator.num_blocks


def test_recompute_prefills_prompt_and_all_generated_tokens():
    sched, a, b, c = preemption_scenario()
    sched.schedule()  # preempts C, which had prompt 4 + 1 generated token
    while c.status is not SequenceStatus.RUNNING:
        batch = fake_step(sched)
        if batch.is_prefill and c in batch.seqs:
            break
    assert c.num_cached_tokens == c.num_tokens - 1  # whole history re-cached, new token appended
    assert len(c.prompt_token_ids) == 4 and len(c.output_token_ids) >= 2


def test_the_newest_sequence_evicts_itself_when_it_alone_needs_a_block():
    sched = make(num_blocks=3, block_size=4)
    a = req(sched, 0, 7, 4)  # 2 blocks; still fits them after its first token (8 tokens)
    b = req(sched, 1, 4, 4)  # 1 block; its first token makes 5 tokens -> needs a 2nd block
    fake_step(sched)
    batch = sched.schedule()  # pool is full and B is both the needy one and the newest
    assert ids(batch.seqs) == [0] and ids(batch.preempted) == [1]
    assert b.status is SequenceStatus.WAITING and b.block_table == [] and ids(sched.waiting) == [1]


def test_older_sequence_evicts_a_newer_one_to_get_a_block():
    sched = make(num_blocks=3, block_size=4)
    a = req(sched, 0, 8, 4)  # 2 blocks; first token makes 9 tokens -> needs a 3rd
    b = req(sched, 1, 4, 4)  # 1 block
    fake_step(sched)
    batch = sched.schedule()  # A needs a block, none free: B (newest) is evicted for it
    assert ids(batch.seqs) == [0] and ids(batch.preempted) == [1]
    assert len(a.block_table) == 3 and b.status is SequenceStatus.WAITING


def test_watermark_holds_back_blocks_only_while_others_are_running():
    sched = make(num_blocks=8, block_size=4, watermark_blocks=2)
    a = req(sched, 0, 12, 4)  # 3 blocks
    fake_step(sched)
    b = req(sched, 1, 16, 4)  # 4 blocks, 5 free: 4 + watermark 2 > 5 -> held back
    batch = fake_step(sched)
    assert not batch.is_prefill and b.status is SequenceStatus.WAITING
    while a.status is not SequenceStatus.FINISHED:
        fake_step(sched)
    # Nothing else running any more: the reserve no longer applies, B is admitted.
    assert fake_step(sched).is_prefill and b.status is SequenceStatus.RUNNING


# ------------------------------------------------------------------- rejection / no deadlock
def test_requests_that_can_never_fit_are_rejected_up_front():
    sched = make(num_blocks=16, block_size=16, max_model_len=1024)
    with pytest.raises(RequestRejected, match="KV blocks"):
        req(sched, 0, 300, 1)  # 300 positions -> 19 blocks > 16
    with pytest.raises(RequestRejected, match="model supports"):
        req(sched, 1, 900, 200)
    assert not sched.waiting and not sched.running  # nothing leaked into the queues

    sched = make(num_blocks=64, block_size=16, max_num_batched_tokens=128)
    with pytest.raises(RequestRejected, match="max_num_batched_tokens"):
        req(sched, 2, 100, 50)  # ctx 149 fits the pool, but a recompute prefill would not fit a step
    assert not sched.waiting and not sched.running


def test_request_using_the_whole_pool_is_admissible_and_completes_among_others():
    sched = make(num_blocks=16, block_size=16, max_num_batched_tokens=1024, watermark_blocks=1)
    big = req(sched, 0, 200, 57)  # ctx = 256 -> exactly 16 blocks
    small = [req(sched, i, 10, 30) for i in range(1, 5)]
    for _ in range(2000):
        if fake_step(sched) is None:
            break
    assert big.status is SequenceStatus.FINISHED and len(big.output_token_ids) == 57
    assert all(s.status is SequenceStatus.FINISHED for s in small)
    assert sched.allocator.num_free_blocks == 16


def test_lone_request_whose_prompt_needs_every_block_is_admitted_despite_the_watermark():
    sched = make(num_blocks=16, block_size=16, max_num_batched_tokens=1024, watermark_blocks=2)
    big = req(sched, 0, 250, 7)  # prompt alone = 16 blocks = the whole pool; 16 + watermark 2 > 16
    batch = sched.schedule()
    assert batch is not None and batch.is_prefill and ids(batch.seqs) == [0]
    assert sched.allocator.num_free_blocks == 0 and big.status is SequenceStatus.RUNNING


def test_max_context_boundary_is_exact():
    sched = make(num_blocks=4, block_size=4)  # 16 tokens of KV
    req(sched, 0, 10, 7)  # ctx = 10 + 7 - 1 = 16 -> fits exactly
    with pytest.raises(RequestRejected):
        req(sched, 1, 10, 8)  # ctx 17 -> 5 blocks


@settings(deadline=None, max_examples=300)
@given(
    workload=st.lists(
        st.tuples(st.integers(1, 40), st.integers(1, 40), st.integers(0, 25)),  # prompt, max_tokens, arrival step
        min_size=1,
        max_size=14,
    ),
    num_blocks=st.integers(3, 20),
    block_size=st.sampled_from([1, 2, 4, 8]),
    max_num_seqs=st.integers(1, 8),
    budget=st.integers(8, 200),
    watermark=st.integers(0, 3),
)
def test_random_workloads_never_deadlock_or_leak(workload, num_blocks, block_size, max_num_seqs, budget, watermark):
    """Whatever the arrivals, pool size and limits: every admissible request finishes with exactly
    max_tokens tokens, rejected ones are exactly the ones that can never fit, the scheduler is never
    stuck while work remains, queues stay in arrival order, and the pool ends full."""
    sched = make(
        num_blocks=num_blocks,
        block_size=block_size,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max(budget, max_num_seqs),
        watermark_blocks=watermark,
    )
    alloc = sched.allocator
    pending = sorted(enumerate(workload), key=lambda x: x[1][2])
    accepted: list[Sequence] = []
    step = 0
    while True:
        while pending and pending[0][1][2] <= step:
            i, (p, m, _) = pending.pop(0)
            seq = Sequence(i, [1] * p, sampling=SamplingParams(max_tokens=m))
            reason = sched.rejection_reason(seq)
            if reason:
                with pytest.raises(RequestRejected):
                    sched.add_request(seq)
            else:
                sched.add_request(seq)
                accepted.append(seq)
        if not pending and not sched.has_unfinished():
            break
        batch = fake_step(sched)
        assert batch is not None or pending, "deadlock: work remains but nothing is schedulable"
        step += 1
        assert step < 50_000, "livelock: no completion"

        # invariants
        if batch is not None:
            assert batch.padded_tokens <= sched.config.max_num_batched_tokens
        assert len(sched.running) <= max_num_seqs
        arrival = [s.seq_id for s in accepted]
        assert ids(sched.running) == sorted(ids(sched.running), key=arrival.index)
        assert ids(sched.waiting) == sorted(ids(sched.waiting), key=arrival.index)
        owned = [b for s in sched.running for b in s.block_table]
        assert len(owned) == len(set(owned)) and alloc.num_free_blocks + len(owned) == alloc.num_blocks
        assert all(not s.block_table for s in sched.waiting)
        for s in sched.running:
            assert len(s.block_table) >= cdiv(s.num_cached_tokens, block_size)

    assert all(s.status is SequenceStatus.FINISHED for s in accepted)
    assert all(len(s.output_token_ids) == s.sampling.max_tokens for s in accepted)
    assert alloc.num_free_blocks == alloc.num_blocks and all(c == 0 for c in alloc.ref_counts)
