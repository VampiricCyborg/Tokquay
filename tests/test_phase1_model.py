"""Phase 1: hand-written GPT-2 matches HuggingFace."""

import copy

import pytest
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

from tokquay.model import GPT2

PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In a shocking finding, scientists discovered a herd of unicorns",
    "def fibonacci(n):\n    if n < 2:\n        return n",
    "Once upon a time, in a land far away,",
    "The capital of France is",
]


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained("gpt2")


@pytest.fixture(scope="module")
def hf():
    return GPT2LMHeadModel.from_pretrained("gpt2").eval()


@pytest.fixture(scope="module")
def ours():
    return GPT2.from_pretrained("gpt2")


@pytest.mark.parametrize("prompt", PROMPTS)
def test_logits_match_hf(prompt, tok, hf, ours):
    ids = tok(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        ref = hf(ids).logits
        got = ours(ids)
    # The brief asks for atol=1e-4, but fp32 logits reach ~320 on some prompts, so
    # accumulated rounding alone gives ~2.4e-4 (HF's own fp32 error vs. float64 is
    # larger than ours). Exactness is proven in the float64 test below; here we
    # only guard against real bugs, which show up as errors >> 1e-3.
    torch.testing.assert_close(got, ref, atol=1e-3, rtol=0)


@pytest.mark.parametrize("prompt", PROMPTS)
def test_logits_match_hf_float64_exact(prompt, tok, hf, ours):
    """In float64 rounding noise vanishes, so any mismatch is a real bug."""
    ids = tok(prompt, return_tensors="pt").input_ids
    hf64 = copy.deepcopy(hf).double()
    ours64 = copy.deepcopy(ours).double()
    with torch.no_grad():
        torch.testing.assert_close(ours64(ids), hf64(ids).logits, atol=1e-8, rtol=0)


@pytest.mark.parametrize("prompt", PROMPTS)
def test_20_greedy_tokens_match_hf(prompt, tok, hf, ours):
    ids = tok(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        ref = hf.generate(
            ids, max_new_tokens=20, do_sample=False,
            pad_token_id=tok.eos_token_id, attention_mask=torch.ones_like(ids),
        )[0, ids.shape[1]:].tolist()

        cur = ids
        got = []
        for _ in range(20):  # no cache: rerun the full sequence every step
            nxt = ours(cur)[:, -1].argmax(-1, keepdim=True)
            got.append(nxt.item())
            cur = torch.cat([cur, nxt], dim=1)
    assert got == ref
