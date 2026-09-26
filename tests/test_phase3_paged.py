"""Phase 3: paged KV cache.

The correctness tests deliberately make paging *hard* to get right:
  * small block sizes, so sequences cross many block boundaries;
  * a shuffled free list, so block tables are non-contiguous and out of order;
  * a pool full of random garbage, so any slot that is read without being masked
    (or written by a padded token) changes the answer.
"""

import copy
import random

import pytest
import torch
from hypothesis import given, settings, strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule
from transformers import AutoTokenizer

from tokquay.generate import greedy_generate, paged_forward, paged_greedy_generate
from tokquay.kv_cache import BlockAllocator, OutOfBlocksError, cdiv
from tokquay.model import GPT2, GPT2Config
from tokquay.sequence import Sequence

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In a shocking finding, scientists discovered a herd of unicorns",
    "def fibonacci(n):\n    if n < 2:\n        return n",
    "Once upon a time, in a land far away,",
    "The capital of France is",
]
LONG_TEXT = (
    "The history of computing is a story of ever smaller machines doing ever larger jobs. "
    "Early computers filled entire rooms, consumed enormous amounts of power, and were "
    "programmed by physically rewiring them. Today a device that fits in a pocket runs "
    "language models with hundreds of millions of parameters, and serving them efficiently "
    "is largely a problem of managing memory: the key value cache grows with every token."
)
TINY = GPT2Config(vocab_size=8, n_positions=64, n_embd=4, n_layer=1, n_head=1)


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained("gpt2")


@pytest.fixture(scope="module")
def model():
    return GPT2.from_pretrained("gpt2").to(DEVICE)


def make_allocator(model, num_blocks, block_size, dtype=torch.float32, cls=BlockAllocator, seed=0):
    """Allocator whose pool holds garbage and whose free list is shuffled."""
    alloc = cls(model.cfg, num_blocks, block_size, device=DEVICE, dtype=dtype)
    g = torch.Generator(device="cpu").manual_seed(seed)
    alloc.pool.copy_((torch.randn(alloc.pool.shape, generator=g) * 50).to(DEVICE, dtype))
    rng = random.Random(seed)
    rng.shuffle(alloc._free)
    return alloc


class RecordingAllocator(BlockAllocator):
    """Remembers every block it ever handed out."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.ever_taken: set[int] = set()

    def _take(self, n):
        blocks = super()._take(n)
        self.ever_taken.update(blocks)
        return blocks


# --------------------------------------------------------------------------- model + paging
@pytest.mark.parametrize("block_size", [4, 16])
@pytest.mark.parametrize("prompt", PROMPTS)
def test_single_sequence_identical_to_phase2(prompt, block_size, tok, model):
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    expected = greedy_generate(model, ids, 24, use_cache=True)  # Phase 2 contiguous cache
    alloc = make_allocator(model, num_blocks=64, block_size=block_size)
    got = paged_greedy_generate(model, alloc, [ids[0].tolist()], 24)[0]
    assert got == expected
    assert alloc.num_free_blocks == alloc.num_blocks


def test_paged_logits_match_full_forward_float64(tok, model):
    """Prefill a prefix, then decode token by token across block boundaries. Every
    step's logits must equal the plain full-sequence forward (float64: noise-free)."""
    m = copy.deepcopy(model).double()
    ids = tok(LONG_TEXT, return_tensors="pt").input_ids[0, :40].tolist()
    with torch.no_grad():
        full = m(torch.tensor([ids], device=DEVICE))[0]  # [40, vocab]
    alloc = make_allocator(m, num_blocks=32, block_size=4, dtype=torch.float64)
    seq = Sequence(0, ids[:6])
    alloc.allocate(seq)
    got = [paged_forward(m, alloc, [seq])[0]]  # prefill positions 0..5 -> logits at position 5
    want_pos = [5]
    for t in range(6, len(ids)):
        seq.append_token(ids[t])
        alloc.append_slot(seq)
        got.append(paged_forward(m, alloc, [seq])[0])  # decode one token at position t
        want_pos.append(t)
    torch.testing.assert_close(torch.stack(got), full[want_pos], atol=1e-9, rtol=0)
    assert seq.num_cached_tokens == len(ids)
    assert len(seq.block_table) == cdiv(len(ids), 4)


def test_batch_of_8_different_lengths_matches_each_alone(tok, model):
    """The brief's headline test: 8 sequences of different lengths in one batch give
    the same output as running each alone (on the Phase 2 contiguous cache)."""
    text_ids = tok(LONG_TEXT, return_tensors="pt").input_ids[0].tolist()
    lengths = [1, 3, 7, 12, 16, 17, 33, 48]  # includes exact block multiples and off-by-one
    prompts = [text_ids[3 * i : 3 * i + n] for i, n in enumerate(lengths)]
    n_new = 20

    alloc = make_allocator(model, num_blocks=128, block_size=16, cls=RecordingAllocator, seed=1)
    before = alloc.pool.clone()
    batched = paged_greedy_generate(model, alloc, prompts, n_new)

    for prompt, got in zip(prompts, batched):
        alone = greedy_generate(model, torch.tensor([prompt], device=DEVICE), n_new, use_cache=True)
        assert got == alone

    assert alloc.num_free_blocks == alloc.num_blocks  # everything returned
    # No block outside the ones handed to sequences was touched (padded tokens must not write).
    changed = {b for b in range(alloc.num_blocks) if not torch.equal(alloc.pool[b], before[b])}
    assert changed <= alloc.ever_taken
    assert alloc.ever_taken != set(range(alloc.num_blocks))  # the test really leaves blocks untouched


def test_batch_shares_blocks_without_overlap(tok, model):
    """Live sequences never share a block, and tables are genuinely non-contiguous."""
    alloc = make_allocator(model, num_blocks=64, block_size=4, seed=2)
    seqs = [Sequence(i, list(range(1, 1 + n))) for i, n in enumerate([5, 9, 13, 2])]
    for s in seqs:
        alloc.allocate(s)
    owned = [b for s in seqs for b in s.block_table]
    assert len(owned) == len(set(owned))
    assert any(b != a + 1 for s in seqs for a, b in zip(s.block_table, s.block_table[1:]))


def test_mixed_prefill_and_decode_in_one_batch_float64(tok, model):
    """One step where sequence A decodes a single token while B prefills its whole
    prompt (different start positions and lengths). Phase 4's scheduler relies on this."""
    m = copy.deepcopy(model).double()
    a_ids = tok(PROMPTS[1], return_tensors="pt").input_ids[0].tolist()
    b_ids = tok(PROMPTS[2], return_tensors="pt").input_ids[0].tolist()
    with torch.no_grad():
        full_a = m(torch.tensor([a_ids], device=DEVICE))[0]
        full_b = m(torch.tensor([b_ids], device=DEVICE))[0]
    alloc = make_allocator(m, num_blocks=64, block_size=4, dtype=torch.float64, seed=3)

    a = Sequence(0, a_ids[:-1])
    alloc.allocate(a)
    paged_forward(m, alloc, [a])  # A prefilled earlier
    a.append_token(a_ids[-1])
    alloc.append_slot(a)
    b = Sequence(1, b_ids)
    alloc.allocate(b)

    logits = paged_forward(m, alloc, [a, b])  # A: 1 new token, B: len(b_ids) new tokens
    torch.testing.assert_close(logits[0], full_a[-1], atol=1e-9, rtol=0)
    torch.testing.assert_close(logits[1], full_b[-1], atol=1e-9, rtol=0)


def test_padding_positions_beyond_n_positions_are_safe(model):
    """A short sequence padded next to one that ends at position 1023 must not index
    wpe out of range, and a sequence that really is too long is rejected."""
    alloc = make_allocator(model, num_blocks=cdiv(1024, 16) + 4, block_size=16)
    long_seq = Sequence(0, [1] * 1023)
    short = Sequence(1, [2, 3])
    alloc.allocate(long_seq)
    alloc.allocate(short)
    paged_forward(model, alloc, [long_seq, short])  # T=1023, short is padded to 1023 slots
    alloc.free(long_seq)
    alloc.free(short)

    too_long = Sequence(2, [1] * 1025)
    alloc.allocate(too_long)
    with pytest.raises(ValueError):
        paged_forward(model, alloc, [too_long])


def test_batch_rejects_block_tables_that_are_too_short(model):
    alloc = make_allocator(model, num_blocks=8, block_size=4)
    s = Sequence(0, [1] * 9)
    alloc.allocate(s, num_tokens=4)  # only one block for 9 tokens
    with pytest.raises(ValueError):
        paged_forward(model, alloc, [s])


# --------------------------------------------------------------------------- allocator
def tiny_alloc(num_blocks=8, block_size=4):
    return BlockAllocator(TINY, num_blocks, block_size, device="cpu")


def test_pool_layout_matches_the_brief(model):
    alloc = tiny_alloc(num_blocks=8, block_size=4)
    assert alloc.pool.shape == (8, TINY.n_layer, 2, TINY.n_head, 4, TINY.d_head)
    full = BlockAllocator(model.cfg, 2, 16, device="cpu")
    assert full.pool.shape == (2, 12, 2, 12, 16, 64)
    assert full.bytes_per_block == 12 * 2 * 12 * 16 * 64 * 4


def test_allocate_takes_ceil_blocks_and_free_returns_them():
    alloc = tiny_alloc()
    s = Sequence(0, [0] * 9)  # 9 tokens / 4 per block -> 3 blocks
    alloc.allocate(s)
    assert len(s.block_table) == 3 and alloc.num_free_blocks == 5
    assert all(alloc.ref_counts[b] == 1 for b in s.block_table)
    alloc.free(s)
    assert s.block_table == [] and alloc.num_free_blocks == 8
    assert all(c == 0 for c in alloc.ref_counts)


def test_append_slot_grabs_a_block_only_when_the_current_one_fills():
    alloc = tiny_alloc(block_size=4)
    s = Sequence(0, [0] * 3)
    alloc.allocate(s)  # 1 block, 1 free slot
    grabbed = []
    for _ in range(9):
        s.append_token(0)
        grabbed.append(alloc.append_slot(s))
    # token counts 4,5,6,7,8,9,10,11,12 -> a new block exactly at 5 and 9
    assert grabbed == [False, True, False, False, False, True, False, False, False]
    assert len(s.block_table) == 3


def test_out_of_blocks_leaves_state_unchanged():
    alloc = tiny_alloc(num_blocks=3, block_size=4)
    s = Sequence(0, [0] * 8)
    alloc.allocate(s)
    big = Sequence(1, [0] * 8)  # needs 2 blocks, only 1 free
    assert not alloc.can_allocate(big.num_tokens)
    with pytest.raises(OutOfBlocksError):
        alloc.allocate(big)
    assert big.block_table == [] and alloc.num_free_blocks == 1

    s.append_token(0)  # 9th token needs a 3rd block: available
    assert alloc.can_append_slot(s)
    alloc.append_slot(s)
    assert alloc.num_free_blocks == 0
    for _ in range(4):
        s.append_token(0)  # 13th token needs a 4th block: none left
    assert not alloc.can_append_slot(s)
    with pytest.raises(OutOfBlocksError):
        alloc.append_slot(s)
    assert len(s.block_table) == 3 and alloc.num_free_blocks == 0


def test_double_allocate_and_double_free_are_caught():
    alloc = tiny_alloc()
    s = Sequence(0, [0] * 4)
    alloc.allocate(s)
    with pytest.raises(ValueError):
        alloc.allocate(s)
    stale = list(s.block_table)
    alloc.free(s)
    alloc.free(s)  # freeing a block-less sequence is a harmless no-op
    s.block_table = stale
    with pytest.raises(RuntimeError):
        alloc.free(s)  # ...but releasing blocks that are already free is a bug


@settings(deadline=None, max_examples=200)
@given(
    sizes=st.lists(st.integers(1, 40), min_size=1, max_size=20),
    order=st.randoms(use_true_random=False),
)
def test_free_count_returns_to_initial_after_any_alloc_free_order(sizes, order):
    alloc = tiny_alloc(num_blocks=16, block_size=4)
    live = []
    for i, n in enumerate(sizes):
        s = Sequence(i, [0] * n)
        if alloc.can_allocate(n):
            alloc.allocate(s)
            live.append(s)
    order.shuffle(live)
    for s in live:
        alloc.free(s)
    assert alloc.num_free_blocks == alloc.num_blocks
    assert all(c == 0 for c in alloc.ref_counts)
    assert sorted(alloc._free) == list(range(alloc.num_blocks))  # every block back, none duplicated


class AllocatorMachine(RuleBasedStateMachine):
    """Random interleavings of allocate / grow / free keep the allocator consistent."""

    def __init__(self):
        super().__init__()
        self.alloc = tiny_alloc(num_blocks=24, block_size=4)
        self.seqs: dict[int, Sequence] = {}
        self.next_id = 0

    @rule(n=st.integers(1, 30))
    def add(self, n):
        s = Sequence(self.next_id, [0] * n)
        self.next_id += 1
        if self.alloc.can_allocate(n):
            self.alloc.allocate(s)
            self.seqs[s.seq_id] = s
        else:
            free_before = self.alloc.num_free_blocks
            with pytest.raises(OutOfBlocksError):
                self.alloc.allocate(s)
            assert self.alloc.num_free_blocks == free_before and s.block_table == []

    @precondition(lambda self: self.seqs)
    @rule(data=st.data())
    def grow(self, data):
        s = self.seqs[data.draw(st.sampled_from(sorted(self.seqs)))]
        s.append_token(0)
        if self.alloc.can_append_slot(s):
            self.alloc.append_slot(s)
        else:
            with pytest.raises(OutOfBlocksError):
                self.alloc.append_slot(s)
            s.output_token_ids.pop()  # the caller would preempt; here we just undo the token

    @precondition(lambda self: self.seqs)
    @rule(data=st.data())
    def free(self, data):
        sid = data.draw(st.sampled_from(sorted(self.seqs)))
        self.alloc.free(self.seqs.pop(sid))

    @invariant()
    def blocks_are_conserved_and_unshared(self):
        owned = [b for s in self.seqs.values() for b in s.block_table]
        assert len(owned) == len(set(owned)), "a block is owned twice"
        assert self.alloc.num_free_blocks + len(owned) == self.alloc.num_blocks
        assert set(owned).isdisjoint(self.alloc._free)
        for b in range(self.alloc.num_blocks):
            assert self.alloc.ref_counts[b] == (1 if b in set(owned) else 0)
        for s in self.seqs.values():
            assert len(s.block_table) == cdiv(s.num_tokens, self.alloc.block_size)

    def teardown(self):
        for s in list(self.seqs.values()):
            self.alloc.free(s)
        assert self.alloc.num_free_blocks == self.alloc.num_blocks


TestAllocatorMachine = AllocatorMachine.TestCase
TestAllocatorMachine.settings = settings(deadline=None, max_examples=100, stateful_step_count=60)
