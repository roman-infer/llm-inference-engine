"""注意力：高性能 SDPA 路径 + 显式 online-softmax 参考路径。"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def attention_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   causal: bool = True, scale: float | None = None) -> torch.Tensor:
    """q,k,v: (B, S, n_h, d_h) -> (B, S, n_h, d_h)。走 Flash/mem-efficient 后端。"""
    # SDPA 期望 (B, n_h, S, d_h)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)
    return out.transpose(1, 2)


def attention_masked(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     attn_mask: torch.Tensor, scale: float) -> torch.Tensor:
    """带自定义 mask 的注意力。

    q: (B, S_q, n_h, dh)；k,v: (B, S_kv, n_h, dh)；
    attn_mask: (S_q, S_kv) bool，True 表示可见（可广播到 B,n_h）。
    返回 (B, S_q, n_h, dh)。
    """
    b, sq, nh, dh = q.shape
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    scores = scores.masked_fill(~attn_mask.view(1, 1, sq, -1), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v)
    return out.transpose(1, 2)


def attention_naive(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    causal: bool = True, scale: float | None = None) -> torch.Tensor:
    """显式实现：物化注意力矩阵，便于逐元素核对。q,k,v: (B,S,n_h,d_h)。"""
    b, s, nh, dh = q.shape
    q = q.transpose(1, 2)  # (B,n_h,S,dh)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    if scale is None:
        scale = 1.0 / dh ** 0.5
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale  # (B,n_h,S,S)
    if causal:
        mask = torch.ones(s, s, device=q.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v)  # (B,n_h,S,dh)
    return out.transpose(1, 2)
