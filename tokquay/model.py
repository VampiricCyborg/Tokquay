"""Hand-written GPT-2 forward pass.

Weights are loaded from a HuggingFace ``GPT2LMHeadModel``; the forward pass
itself is our own. HF stores its projections as ``Conv1D`` with weight shape
``(in, out)``, so we transpose when copying into ``nn.Linear``.

Phase 1: full-sequence forward, no cache (``cache=None``).
Phase 2: optional ``ContiguousKVCache`` splits the forward into prefill (whole
prompt) and decode (one new token attending to the cached K/V).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from tokquay.kv_cache import BatchedKV


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


class ContiguousKVCache:
    """Phase 2 cache: one preallocated tensor each for K and V, sized for ``max_len``.

    Shape ``[n_layer, B, n_head, max_len, d_head]``. ``length`` is how many
    positions are filled; all layers advance together (see ``GPT2.forward``).
    """

    def __init__(self, cfg: GPT2Config, batch_size: int, max_len: int, device=None, dtype=torch.float32):
        shape = (cfg.n_layer, batch_size, cfg.n_head, max_len, cfg.d_head)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.max_len = max_len
        self.length = 0

    def nbytes(self) -> int:
        return (self.k.numel() + self.v.numel()) * self.k.element_size()


KVCache = ContiguousKVCache | BatchedKV  # BatchedKV: PagedBatch, or the baseline's ContiguousBatch


class Attention(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.n_head = cfg.n_head
        self.d_head = cfg.d_head
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)

    def forward(self, x: torch.Tensor, cache: KVCache | None = None, layer: int = 0) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=-1)
        # [B, T, C] -> [B, n_head, T, d_head]
        q, k, v = (t.view(B, T, self.n_head, self.d_head).transpose(1, 2) for t in (q, k, v))

        if isinstance(cache, BatchedKV):
            # Phase 3: scatter new K/V into the block pool, then gather every sequence's
            # blocks back (via its block table) and attend over them in one batched op.
            cache.write(layer, k, v)
            k, v = cache.gather(layer)  # [B, H, nb*block_size, d]
            allowed = cache.mask  # [B, 1, T, nb*block_size]: causal + drops block padding
        else:
            start = 0
            if cache is not None:
                # Write the new K/V into the cache, then attend over everything cached so far.
                start = cache.length
                cache.k[layer, :, :, start : start + T] = k
                cache.v[layer, :, :, start : start + T] = v
                k = cache.k[layer, :, :, : start + T]
                v = cache.v[layer, :, :, : start + T]
            # Query i sits at absolute position start+i and may see keys j <= start+i.
            # Prefill (start=0) gives the usual lower-triangular mask; decode (T=1) sees all.
            q_pos = torch.arange(start, start + T, device=x.device)[:, None]
            k_pos = torch.arange(start + T, device=x.device)[None, :]
            allowed = k_pos <= q_pos

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # [B, H, T, S]
        scores = scores.masked_fill(~allowed, float("-inf"))
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

    def forward(self, x: torch.Tensor, cache: KVCache | None = None, layer: int = 0) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), cache, layer)
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

    def forward(
        self,
        input_ids: torch.Tensor,
        cache: KVCache | None = None,
        last_only: bool = False,
    ) -> torch.Tensor:
        """input_ids: [B, T] -> logits [B, T, vocab] (or [B, 1, vocab] if ``last_only``).

        Without a cache this is the Phase 1 full-sequence forward. With a cache,
        ``input_ids`` are the *new* tokens only: the whole prompt for prefill, or
        a single token per sequence for decode.

        * ``ContiguousKVCache`` (Phase 2): one shared length; advances by T.
        * ``BatchedKV`` (Phase 3 ``PagedBatch``, Phase 6 baseline ``ContiguousBatch``):
          per-sequence start positions and lengths, with ``input_ids`` right-padded
          to T. The caller owns ``num_cached_tokens``. ``last_only`` returns the
          logits of each sequence's last *real* token.
        """
        B, T = input_ids.shape
        paged = isinstance(cache, BatchedKV)
        if paged:
            if (B, T) != (cache.B, cache.T):
                raise ValueError(f"input_ids {tuple(input_ids.shape)} does not match batch {(cache.B, cache.T)}")
            if cache.max_ctx > self.cfg.n_positions:
                raise ValueError(f"sequence length {cache.max_ctx} exceeds n_positions={self.cfg.n_positions}")
            # Real tokens are always in range; only padded slots can overshoot, so clamp them
            # to keep the embedding lookup valid (their outputs are discarded).
            pos = cache.positions.clamp(max=self.cfg.n_positions - 1)  # [B, T]
        else:
            start = cache.length if cache is not None else 0
            if start + T > self.cfg.n_positions:
                raise ValueError(f"sequence length {start + T} exceeds n_positions={self.cfg.n_positions}")
            if cache is not None and start + T > cache.max_len:
                raise ValueError(f"sequence length {start + T} exceeds cache max_len={cache.max_len}")
            pos = torch.arange(start, start + T, device=input_ids.device)

        x = self.wte(input_ids) + self.wpe(pos)
        for i, block in enumerate(self.h):
            x = block(x, cache, i)
        if isinstance(cache, ContiguousKVCache):
            cache.length += T
        if last_only:
            # Skip the [T, vocab] projection for positions we never sample from.
            x = x[torch.arange(B, device=x.device), cache.last_idx][:, None] if paged else x[:, -1:]
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
