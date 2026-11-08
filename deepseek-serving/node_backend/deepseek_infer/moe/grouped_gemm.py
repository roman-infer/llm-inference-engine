"""专家批量算子：token 分组（dispatch）、padded bmm SwiGLU、加权合并（combine）。

torch._grouped_mm 在当前 Windows/sm_89 构建存在限制，故用 padded bmm 自行实现：
将路由到不同专家的 token 按专家分组、pad 到等长，用三次批量矩阵乘完成
SwiGLU FFN（gate/up/down），mask 掉 padding，最后 index_add 加权合并。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Dispatch:
    exp_sorted: torch.Tensor   # (Nk,) 每个 pair 所属专家（按专家排序）
    tok_sorted: torch.Tensor   # (Nk,) 每个 pair 所属 token
    w_sorted: torch.Tensor     # (Nk,) 路由权重
    rank_in_exp: torch.Tensor  # (Nk,) pair 在所属专家内的序号
    padded_tok: torch.Tensor   # (E, max_n) 每个槽位的 token id
    mask: torch.Tensor         # (E, max_n) 有效槽位
    counts: torch.Tensor       # (E,) 每个专家分到的 pair 数


def dispatch(x: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor,
             n_experts: int) -> Dispatch:
    """x (N,d)；indices/weights (N,k)；n_experts 为路由专家总数。"""
    n, k = indices.shape
    nk = n * k
    flat_exp = indices.reshape(nk)
    flat_tok = torch.arange(n, device=x.device).repeat_interleave(k)
    flat_w = weights.reshape(nk)

    order = torch.argsort(flat_exp, stable=True)
    exp_sorted = flat_exp[order]
    tok_sorted = flat_tok[order]
    w_sorted = flat_w[order]

    counts = torch.bincount(flat_exp, minlength=n_experts)
    e_total = n_experts
    offs = torch.zeros(e_total + 1, dtype=torch.long, device=x.device)
    offs[1:] = torch.cumsum(counts, dim=0)
    rank_in_exp = torch.arange(nk, device=x.device) - offs[exp_sorted]

    max_n = int(counts.max().item()) if nk > 0 else 0
    padded_tok = torch.zeros(e_total, max_n, dtype=torch.long, device=x.device)
    mask = torch.zeros(e_total, max_n, dtype=torch.bool, device=x.device)
    padded_tok[exp_sorted, rank_in_exp] = tok_sorted
    mask[exp_sorted, rank_in_exp] = True
    return Dispatch(exp_sorted, tok_sorted, w_sorted, rank_in_exp, padded_tok, mask, counts)


def grouped_swiglu(x: torch.Tensor, d: Dispatch, w_gate: torch.Tensor,
                   w_up: torch.Tensor, w_down: torch.Tensor) -> torch.Tensor:
    """w_gate/w_up: (E,m,d)；w_down: (E,d,m)。返回每 pair 输出 (Nk,d)。"""
    xp = x[d.padded_tok]                       # (E,max_n,d)
    g = torch.bmm(xp, w_gate.transpose(1, 2))  # (E,max_n,m)
    u = torch.bmm(xp, w_up.transpose(1, 2))
    h = torch.nn.functional.silu(g) * u
    o = torch.bmm(h, w_down.transpose(1, 2))    # (E,max_n,d)
    o = o * d.mask.unsqueeze(-1)
    # 恢复到 pair（排序）顺序
    return o[d.exp_sorted, d.rank_in_exp]


def combine(out_pairs: torch.Tensor, d: Dispatch, n_tokens: int) -> torch.Tensor:
    weighted = out_pairs * d.w_sorted.unsqueeze(-1)
    out = torch.zeros(n_tokens, out_pairs.shape[-1], device=out_pairs.device,
                      dtype=out_pairs.dtype)
    out.index_add_(0, d.tok_sorted, weighted)
    return out
