"""DeepSeekMoE 层：top-k 路由专家 + 共享专家。

- routed: gate 线性层选 top-k 专家，选中专家 softmax 归一化加权；
- shared: n_shared 个共享专家对所有 token 生效，各带可学习 gate 标量（se_gate）。
提供 grouped（padded bmm）与 naive（显式逐 token）两条路径用于互相验证。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .grouped_gemm import combine, dispatch, grouped_swiglu
from ..model.norms import swiglu


@dataclass
class MoEConfig:
    d: int
    moe_inter: int
    n_routed: int
    n_active: int
    n_shared: int


class MoE(torch.nn.Module):
    def __init__(self, cfg: MoEConfig, device="cuda", dtype=torch.float16):
        super().__init__()
        self.cfg = cfg
        c = cfg
        std = 0.08
        self.gate = torch.nn.Parameter(torch.empty(c.n_routed, c.d, device=device, dtype=dtype))
        self.w_gate = torch.nn.Parameter(torch.empty(c.n_routed, c.moe_inter, c.d, device=device, dtype=dtype))
        self.w_up = torch.nn.Parameter(torch.empty(c.n_routed, c.moe_inter, c.d, device=device, dtype=dtype))
        self.w_down = torch.nn.Parameter(torch.empty(c.n_routed, c.d, c.moe_inter, device=device, dtype=dtype))
        self.sw_gate = torch.nn.Parameter(torch.empty(c.n_shared, c.moe_inter, c.d, device=device, dtype=dtype))
        self.sw_up = torch.nn.Parameter(torch.empty(c.n_shared, c.moe_inter, c.d, device=device, dtype=dtype))
        self.sw_down = torch.nn.Parameter(torch.empty(c.n_shared, c.d, c.moe_inter, device=device, dtype=dtype))
        self.se_gate = torch.nn.Parameter(torch.ones(c.n_shared, device=device, dtype=dtype))
        for p in self.parameters():
            if p is self.se_gate:
                continue
            torch.nn.init.normal_(p, std=std)

    def _route(self, x: torch.Tensor):
        logits = F.linear(x, self.gate)                 # (N,E)
        vals, idx = torch.topk(logits, self.cfg.n_active, dim=-1)
        weights = torch.softmax(vals, dim=-1)
        return idx, weights

    def _shared(self, x: torch.Tensor) -> torch.Tensor:
        s = self.cfg.n_shared
        X = x.unsqueeze(0).expand(s, -1, -1)            # (S,N,d)
        g = torch.bmm(X, self.sw_gate.transpose(1, 2))
        u = torch.bmm(X, self.sw_up.transpose(1, 2))
        h = F.silu(g) * u
        o = torch.bmm(h, self.sw_down.transpose(1, 2))  # (S,N,d)
        return (o * self.se_gate.reshape(s, 1, 1)).sum(0)

    def forward_grouped(self, x: torch.Tensor) -> torch.Tensor:
        idx, weights = self._route(x)
        d = dispatch(x, idx, weights, self.cfg.n_routed)
        out_pairs = grouped_swiglu(x, d, self.w_gate, self.w_up, self.w_down)
        routed = combine(out_pairs, d, x.shape[0])
        return routed + self._shared(x)

    def forward_naive(self, x: torch.Tensor) -> torch.Tensor:
        idx, weights = self._route(x)
        n = x.shape[0]
        out = torch.zeros(n, self.cfg.d, device=x.device, dtype=x.dtype)
        for i in range(n):
            for j in range(self.cfg.n_active):
                e = int(idx[i, j].item())
                out[i] += weights[i, j] * swiglu(x[i], self.w_gate[e], self.w_up[e],
                                                  self.w_down[e])
            for s in range(self.cfg.n_shared):
                out[i] += self.se_gate[s] * swiglu(x[i], self.sw_gate[s], self.sw_up[s],
                                                    self.sw_down[s])
        return out

    def forward(self, x: torch.Tensor, mode: str = "grouped") -> torch.Tensor:
        return self.forward_grouped(x) if mode == "grouped" else self.forward_naive(x)
