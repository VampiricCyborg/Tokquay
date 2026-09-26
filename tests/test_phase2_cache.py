"""Phase 2: contiguous KV cache gives identical results, and decode stops scaling with length."""

import copy

import pytest
import torch
from transformers import AutoTokenizer

from tokquay.generate import greedy_generate
from tokquay.model import GPT2, ContiguousKVCache

PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In a shocking finding, scientists discovered a herd of unicorns",
    "def fibonacci(n):\n    if n < 2:\n        return n",
    "Once upon a time, in a land far away,",
    "The capital of France is",
]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained("gpt2")


@pytest.fixture(scope="module")
def model():
    return GPT2.from_pretrained("gpt2").to(DEVICE)


@pytest.mark.parametrize("prompt", PROMPTS)
def test_greedy_output_identical_to_phase1(prompt, tok, model):
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    assert greedy_generate(model, ids, 20, use_cache=True) == greedy_generate(model, ids, 20, use_cache=False)


def test_cached_logits_match_full_forward_float64(tok, model):
    """Prefill a prefix, then decode token by token: every step's logits must equal
    the full-sequence forward. float64 removes rounding noise, so 1e-9 is meaningful."""
    m = copy.deepcopy(model).double()
    ids = tok(PROMPTS[1], return_tensors="pt").input_ids.to(DEVICE)
    split = 4
    cache = ContiguousKVCache(m.cfg, 1, ids.shape[1], device=DEVICE, dtype=torch.float64)
    with torch.no_grad():
        full = m(ids)
        got = [m(ids[:, :split], cache)]  # prefill
        for t in range(split, ids.shape[1]):
            got.append(m(ids[:, t : t + 1], cache))  # decode
    torch.testing.assert_close(torch.cat(got, dim=1), full, atol=1e-9, rtol=0)
    assert cache.length == ids.shape[1]


def test_batched_cache_matches_single(tok, model):
    """Equal-length sequences batched together match running each alone."""
    ids = torch.stack([tok(p, return_tensors="pt").input_ids[0, :5] for p in PROMPTS]).to(DEVICE)
    cache = ContiguousKVCache(model.cfg, ids.shape[0], 5 + 10, device=DEVICE)
    with torch.no_grad():
        nxt = model(ids, cache, last_only=True)[:, -1].argmax(-1, keepdim=True)
        batched = [nxt[:, 0]]
        for _ in range(9):
            nxt = model(nxt, cache, last_only=True)[:, -1].argmax(-1, keepdim=True)
            batched.append(nxt[:, 0])
    batched = torch.stack(batched, dim=1).tolist()
    for i in range(ids.shape[0]):
        assert batched[i] == greedy_generate(model, ids[i : i + 1], 10)


def test_cache_overflow_raises(model):
    cache = ContiguousKVCache(model.cfg, 1, 4, device=DEVICE)
    with pytest.raises(ValueError):
        model(torch.zeros(1, 5, dtype=torch.long, device=DEVICE), cache)


def test_decode_time_vs_length_logged(model, capsys):
    """Log per-token decode time as context grows. With a cache a decode step is
    roughly flat in length; without one every step re-runs the whole sequence."""
    rows = []
    for prompt_len in (32, 256, 900):
        ids = torch.randint(0, 50257, (1, prompt_len), device=DEVICE)
        cached: list[float] = []
        uncached: list[float] = []
        greedy_generate(model, ids, 3, use_cache=True)  # warm-up
        greedy_generate(model, ids, 12, use_cache=True, step_times=cached)
        greedy_generate(model, ids, 12, use_cache=False, step_times=uncached)
        rows.append((prompt_len, min(cached) * 1e3, min(uncached) * 1e3))  # min: robust to load spikes
    with capsys.disabled():
        print(f"\n[{DEVICE}] best-case ms per decoded token")
        print("  ctx   cached   no-cache   speedup")
        for n, c, u in rows:
            print(f"  {n:4d}  {c:7.2f}  {u:9.2f}  {u / c:6.1f}x")
    # No-cache cost must grow with context; the cache must win at long context.
    assert rows[-1][2] > rows[0][2]
    assert rows[-1][1] < rows[-1][2]
