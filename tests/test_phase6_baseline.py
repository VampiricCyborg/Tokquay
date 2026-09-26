"""Phase 6: the static-batching baseline is correct, and really is static.

Correctness is the same oracle as everywhere else (each request run alone with the
Phase 2 cache). The static properties are what make it a meaningful baseline: no
request joins a running batch, finished rows keep their slot, and the KV cache is
reserved at ``max_len`` per row up front.
"""

import pytest
import torch

from tokquay.baseline import ContiguousBatch, StaticBatchEngine
from tokquay.scheduler import RequestRejected
from tokquay.sequence import SamplingParams, SequenceStatus

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def greedy(n):
    return SamplingParams(max_tokens=n, ignore_eos=True)


def make(model, batch_size=4, max_len=128):
    return StaticBatchEngine(model, batch_size, max_len, device=DEVICE)


def test_mixed_batch_matches_reference_including_rows_that_finish_early(shared_model, text_ids, greedy_reference):
    specs = [  # (prompt, max_tokens): different prompt lengths, and rows that end at different steps
        (text_ids[0:5], 1),  # done at the prefill step, then just occupies its row
        (text_ids[10:40], 12),
        (text_ids[50:58], 25),
        (text_ids[70:71], 6),
        (text_ids[100:133], 18),
    ]
    engine = make(shared_model, batch_size=8)
    seqs = [engine.add_request(p, greedy(n)) for p, n in specs]
    engine.run_until_done()
    for seq, (prompt, n) in zip(seqs, specs):
        assert seq.output_token_ids == greedy_reference(prompt, n)
        assert seq.finish_reason == "length" and seq.status is SequenceStatus.FINISHED
    assert engine.num_batches == 1 and engine.stats.max_batch == 5
    assert engine.stats.prefill_steps == 1 and engine.stats.decode_steps == 24  # the slowest row (25 tokens) sets the length


def test_a_request_arriving_mid_batch_waits_for_the_whole_batch_to_drain(shared_model, text_ids, greedy_reference):
    engine = make(shared_model, batch_size=4)
    first = [engine.add_request(text_ids[i * 20 : i * 20 + 10], greedy(n)) for i, n in enumerate((6, 30, 12))]
    for _ in range(3):
        engine.step()
    late = engine.add_request(text_ids[100:110], greedy(5))  # arrives while the batch is decoding
    assert engine.batch and len(engine.batch) == 3

    first_ids = {s.seq_id for s in first}
    steps, last_of_first = 0, None
    while True:
        ids = {e.seq_id for e in engine.step()}
        steps += 1
        if late.seq_id in ids:
            break  # the late request's own prefill
        assert ids <= first_ids  # until then, only the first batch's rows produce tokens
        last_of_first = steps
    # Even though the 6-token and 12-token requests were long finished, the late one only
    # got going after the 30-token request drained the batch. That batch takes 30 steps in all
    # (a prefill and 29 decodes), 3 are already done, so 27 remain, then the late prefill.
    assert all(s.status is SequenceStatus.FINISHED for s in first)
    assert last_of_first == steps - 1
    assert steps == (30 - 3) + 1

    engine.run_until_done()
    assert late.output_token_ids == greedy_reference(text_ids[100:110], 5)
    assert engine.num_batches == 2


def test_batches_form_fcfs_up_to_the_batch_size(shared_model, text_ids):
    engine = make(shared_model, batch_size=4)
    seqs = [engine.add_request(text_ids[i * 7 : i * 7 + 6], greedy(3)) for i in range(10)]
    first_step_of = {}
    step = 0
    while engine.has_unfinished():
        step += 1
        for e in engine.step():
            first_step_of.setdefault(e.seq_id, step)
    batches = {}
    for s in seqs:
        batches.setdefault(first_step_of[s.seq_id], []).append(s.seq_id)
    assert list(batches.values()) == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]  # arrival order, 4 at a time
    assert engine.num_batches == 3


def test_kv_cache_is_reserved_at_max_len_per_row_whatever_the_requests_need(shared_model, text_ids):
    engine = make(shared_model, batch_size=4, max_len=128)
    assert engine.kv_snapshot().allocated_tokens == 0  # nothing is running yet
    engine.add_request(text_ids[:5], greedy(3))
    engine.add_request(text_ids[10:20], greedy(3))
    engine.step()
    snap = engine.kv_snapshot()
    assert snap.num_seqs == 2 and snap.allocated_tokens == 2 * 128  # slabs for 2 rows, though 15 tokens are cached
    assert snap.live_tokens == 5 + 10
    engine.run_until_done()
    assert engine.kv_snapshot().allocated_tokens == 0  # the batch drained, the rows are free again
    cfg = shared_model.cfg
    assert engine.bytes_per_token == 2 * cfg.n_layer * cfg.n_head * cfg.d_head * 4
    assert engine.k.numel() * 4 * 2 == 4 * 128 * engine.bytes_per_token  # the whole reservation exists up front


def test_unservable_requests_are_rejected_up_front(shared_model, text_ids):
    engine = make(shared_model, batch_size=2, max_len=64)
    with pytest.raises(RequestRejected):
        engine.add_request(text_ids[:40], greedy(25))  # 40 + 25 > 64
    engine.add_request(text_ids[:40], greedy(24))  # exactly max_len is fine
    with pytest.raises(ValueError):
        engine.add_request(text_ids[:5], SamplingParams(max_tokens=3, temperature=0.8))  # greedy only
    with pytest.raises(ValueError):
        StaticBatchEngine(shared_model, 2, max_len=shared_model.cfg.n_positions + 1)


def test_max_len_row_can_be_used_to_the_last_slot(shared_model, text_ids, greedy_reference):
    """A request that fills its row exactly, next to one that finishes early and keeps
    getting dummy tokens written into its own row: neither may touch the other."""
    engine = make(shared_model, batch_size=2, max_len=64)
    long = engine.add_request(text_ids[:30], greedy(34))  # 30 + 34 = 64
    short = engine.add_request(text_ids[40:60], greedy(2))
    engine.run_until_done()
    assert long.output_token_ids == greedy_reference(text_ids[:30], 34)
    assert short.output_token_ids == greedy_reference(text_ids[40:60], 2)


def test_contiguous_batch_validates_its_inputs():
    k = torch.zeros(2, 3, 2, 16, 4)
    with pytest.raises(ValueError):
        ContiguousBatch(k, k, [0] * 4, [1] * 4)  # more rows than the cache has
    with pytest.raises(ValueError):
        ContiguousBatch(k, k, [10], [7])  # needs 17 slots, max_len is 16
    with pytest.raises(ValueError):
        ContiguousBatch(k, k, [0], [0])  # a row must feed at least one token
