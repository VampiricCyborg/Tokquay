"""Shared, session-scoped fixtures so the GPT-2 weights load once for the engine tests."""

import pytest
import torch
from transformers import AutoTokenizer

from tokquay.model import GPT2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_TEXT = (
    "The history of computing is a story of ever smaller machines doing ever larger jobs. "
    "Early computers filled entire rooms, consumed enormous amounts of power, and were "
    "programmed by physically rewiring them. Today a device that fits in a pocket runs "
    "language models with hundreds of millions of parameters, and serving them efficiently "
    "is largely a problem of managing memory: the key value cache grows with every token. "
)


@pytest.fixture(scope="session")
def shared_tok():
    return AutoTokenizer.from_pretrained("gpt2")


@pytest.fixture(scope="session")
def shared_model():
    return GPT2.from_pretrained("gpt2").to(DEVICE)


@pytest.fixture(scope="session")
def text_ids(shared_tok):
    """~500 real-text token ids to slice prompts of any length out of."""
    return shared_tok(_TEXT * 6).input_ids
