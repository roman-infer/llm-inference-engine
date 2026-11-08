"""RMSNorm 与公共 FFN 算子。"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def swiglu(x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor,
           w_down: torch.Tensor) -> torch.Tensor:
    """SwiGLU FFN。w_gate/w_up (m,d), w_down (d,m)。"""
    h = F.silu(F.linear(x, w_gate)) * F.linear(x, w_up)
    return F.linear(h, w_down)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, device="cuda", dtype=torch.float16):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight
