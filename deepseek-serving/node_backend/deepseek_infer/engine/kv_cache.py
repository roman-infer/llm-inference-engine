"""MLA latent KV cache：每层缓存压缩潜变量 c_kv 与解耦 RoPE key。"""
from __future__ import annotations

import torch

from ..model.config import ModelConfig


class LatentKVCache:
    def __init__(self, cfg: ModelConfig, device="cuda", dtype=torch.float16):
        L = cfg.n_layers
        T = cfg.max_seq_len
        self.ckv = torch.zeros(L, T, cfg.d_latent, device=device, dtype=dtype)
        self.krope = torch.zeros(L, T, cfg.n_heads, cfg.d_head_rope, device=device, dtype=dtype)
        self.seq_len = 0

    def set_layer(self, layer: int, ckv: torch.Tensor, krope: torch.Tensor, start: int = 0):
        s = ckv.shape[0]
        self.ckv[layer, start:start + s].copy_(ckv)
        self.krope[layer, start:start + s].copy_(krope)
        self.seq_len = max(self.seq_len, start + s)

    def set_token(self, layer: int, position: int, ckv: torch.Tensor, krope: torch.Tensor):
        self.ckv[layer, position].copy_(ckv)
        self.krope[layer, position].copy_(krope)
        self.seq_len = max(self.seq_len, position + 1)

    def truncate(self, position: int):
        """丢弃 >= position 的缓存（投机验证拒绝后回滚）。"""
        self.ckv[:, position:].zero_()
        self.krope[:, position:].zero_()
        self.seq_len = position

    def reset(self):
        self.ckv.zero_()
        self.krope.zero_()
        self.seq_len = 0

    def bytes_per_token(self, dtype_bytes: int = 2) -> int:
        return (self.ckv.shape[-1] + self.krope.shape[-1] * self.krope.shape[-2]) * dtype_bytes