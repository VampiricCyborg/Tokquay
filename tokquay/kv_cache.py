"""Phase 3: paged KV cache.

KV memory is one preallocated pool of fixed-size blocks::

    pool: [num_blocks, n_layer, 2, n_head, block_size, d_head]      (dim 2: 0=K, 1=V)

A block holds ``block_size`` consecutive token positions for *all* layers of
one sequence. Each sequence owns a ``block_table`` (a list of block ids), the
same way a process owns a page table: logical position ``p`` lives in block
``block_table[p // block_size]`` at offset ``p % block_size``. Blocks are handed
out on demand, so nothing is reserved for the maximum length up front.

``BlockAllocator`` owns the pool, the free list and the ref counts.
``PagedBatch`` is the per-step view the model uses: it turns block tables into
the index tensors for scattering new K/V into the pool and gathering it back.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tokquay.sequence import Sequence

if TYPE_CHECKING:  # avoid a runtime import cycle: model.py imports PagedBatch from here
    from tokquay.model import GPT2Config


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


class OutOfBlocksError(RuntimeError):
    """Raised (without changing any state) when the free list is too short."""


class BlockAllocator:
    def __init__(
        self,
        cfg: "GPT2Config",
        num_blocks: int,
        block_size: int = 16,
        device=None,
        dtype: torch.dtype = torch.float32,
    ):
        if num_blocks < 1 or block_size < 1:
            raise ValueError("num_blocks and block_size must be positive")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.pool = torch.zeros(
            num_blocks, cfg.n_layer, 2, cfg.n_head, block_size, cfg.d_head, device=device, dtype=dtype
        )
        # LIFO free list; reversed so block 0 is handed out first.
        self._free: list[int] = list(range(num_blocks - 1, -1, -1))
        self.ref_counts: list[int] = [0] * num_blocks

    # ---- bookkeeping ---------------------------------------------------------------
    @property
    def num_free_blocks(self) -> int:
        return len(self._free)

    @property
    def num_used_blocks(self) -> int:
        return self.num_blocks - len(self._free)

    @property
    def bytes_per_block(self) -> int:
        return self.pool[0].numel() * self.pool.element_size()

    def blocks_needed(self, num_tokens: int) -> int:
        return cdiv(num_tokens, self.block_size)

    # ---- allocation ----------------------------------------------------------------
    def can_allocate(self, num_tokens: int) -> bool:
        return self.blocks_needed(num_tokens) <= self.num_free_blocks

    def allocate(self, seq: Sequence, num_tokens: int | None = None) -> None:
        """Give a block-less sequence enough blocks for ``num_tokens`` positions
        (default: every token it currently has, prompt + any recomputed output)."""
        if seq.block_table:
            raise ValueError(f"sequence {seq.seq_id} already owns blocks")
        n = self.blocks_needed(seq.num_tokens if num_tokens is None else num_tokens)
        seq.block_table = self._take(n)

    def can_append_slot(self, seq: Sequence) -> bool:
        return self.blocks_needed(seq.num_tokens) - len(seq.block_table) <= self.num_free_blocks

    def append_slot(self, seq: Sequence) -> bool:
        """Make sure the block table covers ``seq.num_tokens`` positions.

        Call after appending a sampled token and before the decode step that
        will write its K/V. Grabs a new block only when the current one is full;
        returns True if it did. Raises ``OutOfBlocksError`` (state unchanged) if
        the pool is exhausted, which is the scheduler's cue to preempt.
        """
        missing = self.blocks_needed(seq.num_tokens) - len(seq.block_table)
        if missing <= 0:
            return False
        seq.block_table.extend(self._take(missing))
        return True

    def free(self, seq: Sequence) -> None:
        """Release all of the sequence's blocks (ref-counted; safe to call twice)."""
        for b in seq.block_table:
            if self.ref_counts[b] <= 0:
                raise RuntimeError(f"double free of block {b}")
            self.ref_counts[b] -= 1
            if self.ref_counts[b] == 0:
                self._free.append(b)
        seq.block_table = []

    def _take(self, n: int) -> list[int]:
        if n > len(self._free):
            raise OutOfBlocksError(f"need {n} blocks, only {len(self._free)} free")
        blocks = [self._free.pop() for _ in range(n)]
        for b in blocks:
            self.ref_counts[b] = 1
        return blocks

    # ---- model-facing --------------------------------------------------------------
    def batch_for(self, seqs: list[Sequence], new_lens: list[int] | None = None) -> "PagedBatch":
        """Build the step view for ``seqs``: each feeds ``token_ids[num_cached_tokens:]``
        (or just ``new_lens[i]`` tokens of it) starting at position ``num_cached_tokens``."""
        starts = [s.num_cached_tokens for s in seqs]
        new_lens = new_lens or [s.num_uncached_tokens for s in seqs]
        return PagedBatch(self.pool, self.block_size, [s.block_table for s in seqs], starts, new_lens)


class BatchedKV:
    """What ``GPT2.forward`` needs from a per-step batch view of the KV cache.

    ``PagedBatch`` (below) keeps K/V in a block pool addressed through block tables;
    ``baseline.ContiguousBatch`` keeps each row's K/V in its own pre-reserved slab.
    The model does not care which: it only uses these members.

    * ``B``, ``T``: batch size and (padded) number of new tokens per row.
    * ``max_ctx``: longest row after this step, as a Python int.
    * ``positions`` ``[B, T]``: absolute position of every new token.
    * ``last_idx`` ``[B]``: column of each row's last real token.
    * ``mask`` ``[B, 1, T, S]``: True where a query may attend to a key slot.
    * ``write(layer, k, v)``: store new K/V ``[B, H, T, d]``; padded slots must not write.
    * ``gather(layer)``: every row's K and V as ``[B, H, S, d]`` (``S`` matches ``mask``).
    """

    B: int
    T: int
    max_ctx: int
    positions: torch.Tensor
    last_idx: torch.Tensor
    mask: torch.Tensor

    def write(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> None:
        raise NotImplementedError

    def gather(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class PagedBatch(BatchedKV):
    """Everything the model needs to run one batched step over paged KV.

    Sequence ``b`` feeds ``new_lens[b]`` new tokens at absolute positions
    ``start[b] ... start[b]+new_lens[b]-1``; its K/V for earlier positions is
    already in the pool. Inputs are right-padded to ``T = max(new_lens)``, and
    padded slots never write to the pool. Because ``start`` and ``new_lens``
    are per-sequence, prefill (start=0, many tokens), decode (one token) and a
    mix of both fit in one batch.

    All index tensors are built once here from Python lists (no GPU syncs) and
    reused by every layer.
    """

    def __init__(
        self,
        pool: torch.Tensor,
        block_size: int,
        block_tables: list[list[int]],
        start: list[int],
        new_lens: list[int],
    ):
        if not block_tables or not (len(block_tables) == len(start) == len(new_lens)):
            raise ValueError("block_tables, start and new_lens must be non-empty and the same length")
        if min(new_lens) < 1:
            raise ValueError("every sequence must feed at least one token")
        device = pool.device
        bs = block_size
        ctx = [s + n for s, n in zip(start, new_lens)]  # keys visible after this step
        for b, (tbl, c) in enumerate(zip(block_tables, ctx)):
            if len(tbl) < cdiv(c, bs):
                raise ValueError(f"sequence {b}: {len(tbl)} blocks cannot hold {c} tokens")

        B, T = len(block_tables), max(new_lens)
        nb = max(cdiv(c, bs) for c in ctx)  # blocks to gather (longest sequence)
        self.pool, self.block_size, self.B, self.T, self.num_blocks_gathered = pool, bs, B, T, nb
        self.max_ctx = max(ctx)  # longest sequence after this step, as a Python int

        # Block tables padded to nb with block 0. Padding is masked out below, so
        # what block 0 happens to contain never affects the result.
        padded = [tbl[:nb] + [0] * (nb - len(tbl[:nb])) for tbl in block_tables]
        self.block_tables = torch.tensor(padded, dtype=torch.long, device=device)

        start_t = torch.tensor(start, dtype=torch.long, device=device)
        ctx_t = torch.tensor(ctx, dtype=torch.long, device=device)
        self.positions = start_t[:, None] + torch.arange(T, device=device)[None, :]  # [B, T]
        self.last_idx = torch.tensor([n - 1 for n in new_lens], dtype=torch.long, device=device)

        # allowed[b, 0, t, j]: query (b, t) at position start+t may attend to key j iff
        # j <= start+t (causal). For a real token this also hides the unused slots of its
        # last block and any block-table padding, since start+t < ctx. The extra j < ctx
        # term only matters for padded query rows (start+t >= ctx): it keeps them off
        # unwritten slots, though their outputs are discarded either way.
        k_pos = torch.arange(nb * bs, device=device)[None, None, :]
        self.mask = ((k_pos <= self.positions[:, :, None]) & (k_pos < ctx_t[:, None, None]))[:, None]

        # Scatter targets for the *valid* new tokens only: (batch row, token col, block, offset).
        w_b, w_t, w_blk, w_off = [], [], [], []
        for b, (tbl, s, n) in enumerate(zip(block_tables, start, new_lens)):
            for t in range(n):
                p = s + t
                w_b.append(b)
                w_t.append(t)
                w_blk.append(tbl[p // bs])
                w_off.append(p % bs)
        w = torch.tensor([w_b, w_t, w_blk, w_off], dtype=torch.long, device=device)
        self._w_b, self._w_t, self._w_blk, self._w_off = w

    def write(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Scatter new K/V ([B, H, T, d]) into the pool at each token's (block, offset)."""
        pl = self.pool[:, layer]  # view: [num_blocks, 2, H, block_size, d]
        pl[self._w_blk, 0, :, self._w_off] = k[self._w_b, :, self._w_t]  # -> [N, H, d]
        pl[self._w_blk, 1, :, self._w_off] = v[self._w_b, :, self._w_t]

    def gather(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather each sequence's K and V as contiguous [B, H, nb*block_size, d]."""
        B = self.B
        out = []
        for kv in (0, 1):
            g = self.pool[self.block_tables, layer, kv]  # [B, nb, H, block_size, d]
            g = g.permute(0, 2, 1, 3, 4)  # [B, H, nb, block_size, d]
            out.append(g.reshape(B, g.shape[1], -1, g.shape[-1]))
        return out[0], out[1]
