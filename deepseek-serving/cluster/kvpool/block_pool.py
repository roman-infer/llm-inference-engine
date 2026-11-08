"""块式 KV 池：在 MLA latent KV 之上做物理块池、块表、引用计数与前缀共享（CoW）。

与节点后端 LatentKVCache 的关系：latent KV 的**单元布局**（ckv + krope）不变，
但节点后端那个"每序列一条稠密预分配 tensor"被这里的"物理块池 + 每序列块表"取代，
从而支持跨请求前缀复用、分层卸载与跨引擎传输。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from node_backend.deepseek_infer.model.config import ModelConfig


@dataclass
class KVBlock:
    block_id: int
    ckv: torch.Tensor | None        # (L, B, d_latent)；在 ssd 层时为 None
    krope: torch.Tensor | None      # (L, B, n_heads, d_head_rope)
    ckv_shape: tuple
    krope_shape: tuple
    fill: int = 0
    refs: int = 0
    tier: str = "gpu"
    last_used: int = 0


class SequenceKVMeta:
    """一条逻辑序列的块表：logical block index -> physical block id。"""

    def __init__(self, seq_id: str, pool: "BlockPool"):
        self.seq_id = seq_id
        self.pool = pool
        self.table: dict[int, int] = {}
        self.length = 0

    def physical_ids(self) -> list[int]:
        return [self.table[j] for j in sorted(self.table)]


class BlockPool:
    def __init__(self, cfg: ModelConfig, device="cpu", dtype=torch.float16,
                 block_size: int = 16):
        self.L = cfg.n_layers
        self.B = block_size
        self.dc = cfg.d_latent
        self.nh = cfg.n_heads
        self.dr = cfg.d_head_rope
        self.device = device
        self.dtype = dtype
        self.blocks: dict[int, KVBlock] = {}
        self._free: list[int] = []
        self._next_id = 1
        self.clock = 0

    def _mk(self, bid: int) -> KVBlock:
        ckv_shape = (self.L, self.B, self.dc)
        krope_shape = (self.L, self.B, self.nh, self.dr)
        return KVBlock(
            bid,
            torch.zeros(*ckv_shape, device=self.device, dtype=self.dtype),
            torch.zeros(*krope_shape, device=self.device, dtype=self.dtype),
            ckv_shape, krope_shape,
        )

    def alloc(self, refs: int = 1) -> KVBlock:
        if self._free:
            bid = self._free.pop()
            b = self.blocks[bid]
            if b.ckv is None:
                b.ckv = torch.zeros(*b.ckv_shape, device=self.device, dtype=self.dtype)
                b.krope = torch.zeros(*b.krope_shape, device=self.device, dtype=self.dtype)
                b.tier = "gpu"
        else:
            bid = self._next_id
            self._next_id += 1
            b = self._mk(bid)
            self.blocks[bid] = b
        b.refs = refs
        b.fill = 0
        return b

    def retain(self, bid: int) -> KVBlock:
        b = self.blocks[bid]
        b.refs += 1
        b.last_used = self.clock
        self.clock += 1
        return b

    def release(self, bid: int) -> None:
        b = self.blocks[bid]
        b.refs -= 1
        if b.refs == 0:
            if b.ckv is not None:
                b.ckv.zero_()
                b.krope.zero_()
            b.fill = 0
            self._free.append(bid)

    # ---------- 写入 ----------
    def append_tokens(self, meta: SequenceKVMeta,
                      ckv_tok: torch.Tensor, krope_tok: torch.Tensor) -> None:
        T = ckv_tok.shape[1]
        for t in range(T):
            logical = (meta.length + t) // self.B
            off = (meta.length + t) % self.B
            if logical not in meta.table:
                meta.table[logical] = self.alloc(1).block_id
            bid = meta.table[logical]
            b = self.blocks[bid]
            if b.refs > 1:                          # 写共享块 -> CoW
                nb = self.alloc(1)
                if b.ckv is not None:
                    nb.ckv.copy_(b.ckv)
                    nb.krope.copy_(b.krope)
                    nb.fill = b.fill
                meta.table[logical] = nb.block_id
                self.release(bid)
                b = nb
            b.ckv[:, off].copy_(ckv_tok[:, t])
            b.krope[:, off].copy_(krope_tok[:, t])
            b.fill = max(b.fill, off + 1)
            b.last_used = self.clock
            self.clock += 1
        meta.length += T

    # ---------- 前缀共享 ----------
    def share_prefix(self, src: SequenceKVMeta, dst: SequenceKVMeta,
                     n_blocks: int, shared_tokens: int) -> None:
        for j in range(n_blocks):
            bid = src.table[j]
            dst.table[j] = self.retain(bid).block_id
        dst.length = shared_tokens

    # ---------- 导出 / 导入（供 Connector）----------
    def to_block_data(self, meta: SequenceKVMeta, logical_index: int):
        from cluster.comm.kv_connector import BlockData
        b = self.blocks[meta.table[logical_index]]
        assert b.ckv is not None, "block is offloaded; load before export"
        return BlockData(
            logical_index,
            b.ckv[:, :b.fill].clone(),
            b.krope[:, :b.fill].clone(),
            b.fill,
        )

    def import_block_data(self, meta: SequenceKVMeta, bd) -> None:
        b = self.alloc(1)
        b.ckv[:, :bd.fill].copy_(bd.ckv)
        b.krope[:, :bd.fill].copy_(bd.krope)
        b.fill = bd.fill
        meta.table[bd.logical_index] = b.block_id
        meta.length = max(meta.length, bd.logical_index * self.B + bd.fill)

    def free_sequence(self, meta: SequenceKVMeta) -> None:
        for bid in set(meta.table.values()):
            self.release(bid)
        meta.table.clear()
        meta.length = 0
