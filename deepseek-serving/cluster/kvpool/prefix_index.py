"""全局前缀索引：按块边界对 token 前缀做确定性哈希，支持最长前缀匹配。

命中后配合 BlockPool.share_prefix 复用物理块（全局 KV 复用）。
"""
from __future__ import annotations

from typing import Optional

import torch

_BASE = 1_000_003
_MASK = (1 << 64) - 1


def _update(h: int, block_tokens) -> int:
    for tok in block_tokens.tolist():
        h = (h * _BASE + int(tok)) & _MASK
    return h


class PrefixIndex:
    def __init__(self, block_size: int):
        self.B = block_size
        self._map: dict[int, tuple] = {}

    def register(self, tokens: torch.Tensor, meta, n_blocks: Optional[int] = None) -> None:
        total = tokens.shape[0]
        nb = n_blocks if n_blocks is not None else total // self.B
        h = 0
        for j in range(nb):
            h = _update(h, tokens[j * self.B:(j + 1) * self.B])
            self._map[h] = (meta, j + 1)

    def match(self, tokens: torch.Tensor):
        nb = tokens.shape[0] // self.B
        h = 0
        best = None
        for j in range(nb):
            h = _update(h, tokens[j * self.B:(j + 1) * self.B])
            if h in self._map:
                best = self._map[h]
        return best                    # (meta, n_blocks) 或 None
