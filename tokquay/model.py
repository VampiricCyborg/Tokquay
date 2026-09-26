"""Hand-written GPT-2 forward pass (Phase 1: no KV cache).

Weights are loaded from a HuggingFace ``GPT2LMHeadModel``; the forward pass
itself is our own. HF stores its projections as ``Conv1D`` with weight shape
``(in, out)``, so we transpose when copying into ``nn.Linear``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPT2Config:
    vocab_size: int = 50257
    n_positions: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    layer_norm_eps: float = 1e-5

    @property
    def d_head(self) -> int:
        return self.n_embd // self.n_head


class Attention(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.n_head = cfg.n_head
        self.d_head = cfg.d_head
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=-1)
        # [B, T, C] -> [B, n_head, T, d_head]
        q, k, v = (t.view(B, T, self.n_head, self.d_head).transpose(1, 2) for t in (q, k, v))

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # [B, H, T, T]
        causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        scores = scores.masked_fill(~causal, float("-inf"))
        out = F.softmax(scores, dim=-1) @ v  # [B, H, T, d_head]

        out = out.transpose(1, 2).reshape(B, T, C)
        return self.c_proj(out)


class MLP(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # GPT-2 uses the tanh approximation of GELU ("gelu_new" in HF).
        return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))


class Block(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_eps)
        self.attn = Attention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(nn.Module):
    def __init__(self, cfg: GPT2Config | None = None):
        super().__init__()
        self.cfg = cfg or GPT2Config()
        self.wte = nn.Embedding(self.cfg.vocab_size, self.cfg.n_embd)
        self.wpe = nn.Embedding(self.cfg.n_positions, self.cfg.n_embd)
        self.h = nn.ModuleList(Block(self.cfg) for _ in range(self.cfg.n_layer))
        self.ln_f = nn.LayerNorm(self.cfg.n_embd, eps=self.cfg.layer_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids: [B, T] -> logits [B, T, vocab]."""
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device)
        x = self.wte(input_ids) + self.wpe(pos)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        return x @ self.wte.weight.T  # tied LM head

    @classmethod
    def from_pretrained(cls, name: str = "gpt2") -> "GPT2":
        from transformers import GPT2LMHeadModel

        hf = GPT2LMHeadModel.from_pretrained(name)
        hc = hf.config
        cfg = GPT2Config(
            vocab_size=hc.vocab_size,
            n_positions=hc.n_positions,
            n_embd=hc.n_embd,
            n_layer=hc.n_layer,
            n_head=hc.n_head,
            layer_norm_eps=hc.layer_norm_epsilon,
        )
        model = cls(cfg)
        sd = hf.state_dict()
        # Some transformers versions prefix keys with "transformer.", handle both.
        get = lambda k: sd[k] if k in sd else sd[f"transformer.{k}"]  # noqa: E731

        with torch.no_grad():
            model.wte.weight.copy_(get("wte.weight"))
            model.wpe.weight.copy_(get("wpe.weight"))
            model.ln_f.weight.copy_(get("ln_f.weight"))
            model.ln_f.bias.copy_(get("ln_f.bias"))
            for i, blk in enumerate(model.h):
                p = f"h.{i}."
                for ln in ("ln_1", "ln_2"):
                    getattr(blk, ln).weight.copy_(get(f"{p}{ln}.weight"))
                    getattr(blk, ln).bias.copy_(get(f"{p}{ln}.bias"))
                for mod, name_ in (
                    (blk.attn.c_attn, "attn.c_attn"),
                    (blk.attn.c_proj, "attn.c_proj"),
                    (blk.mlp.c_fc, "mlp.c_fc"),
                    (blk.mlp.c_proj, "mlp.c_proj"),
                ):
                    mod.weight.copy_(get(f"{p}{name_}.weight").t())  # Conv1D is (in, out)
                    mod.bias.copy_(get(f"{p}{name_}.bias"))
        return model.eval()
