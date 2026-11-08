"""RoPE：支持 MLA 解耦 RoPE（只旋转 q/k 的 rope 子维度，d_h^R）。

采用 GPT-J 风格（前半/后半成对旋转），与 DeepSeek 官方实现一致。
"""
from __future__ import annotations

import torch


def precompute_rope(dim: int, max_seq_len: int, base: float = 10000.0,
                    device="cuda", dtype=torch.float32):
    """返回 cos/sin，shape (max_seq_len, dim//2)。"""
    assert dim % 2 == 0
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim))
    t = torch.arange(max_seq_len, device=device, dtype=dtype)
    angles = torch.einsum("i,j->ij", t, inv_freq)  # (T, dim/2)
    return angles.cos(), angles.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
               positions: torch.Tensor | None = None) -> torch.Tensor:
    """对 x 施加 RoPE。

    x: (B, S, n_h, dim) 或 (S, n_h, dim)；
    positions: 自定义位置（如 decode 传入 [pos]），默认按序列顺序 0..S-1。
    cos/sin: (max_seq, dim//2)。
    """
    squeeze = False
    if x.dim() == 3:
        x = x.unsqueeze(0)  # (1,S,n_h,dim)
        squeeze = True
    b, s, nh, dim = x.shape
    if positions is None:
        positions = torch.arange(s, device=x.device)
    c = cos[positions]  # (S,dim/2)
    si = sin[positions]
    # 广播到 (1,S,1,dim/2)
    c = c.view(1, s, 1, dim // 2)
    si = si.view(1, s, 1, dim // 2)
    xf = x.float()
    out = xf[..., : dim // 2] * c - xf[..., dim // 2:] * si
    out2 = xf[..., : dim // 2] * si + xf[..., dim // 2:] * c
    rotated = torch.cat((out, out2), dim=-1).to(x.dtype)
    if squeeze:
        rotated = rotated.squeeze(0)
    return rotated
