"""测试公共工具。"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch  # noqa: E402


def rel_err(actual: torch.Tensor, expected: torch.Tensor) -> float:
    a = actual.detach().float()
    e = expected.detach().float()
    return ((a - e).abs() / (e.abs() + 1e-6)).mean().item()


def frobenius_rel_err(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """全局相对误差 ||a-e||_F / ||e||_F（零均值输出的正确度量）。"""
    a = actual.detach().float()
    e = expected.detach().float()
    return (a - e).norm().item() / e.norm().clamp_min(1e-8).item()


def cosine_sim(actual: torch.Tensor, expected: torch.Tensor) -> float:
    a = actual.detach().float().flatten()
    e = expected.detach().float().flatten()
    return torch.nn.functional.cosine_similarity(a, e, dim=0).item()


def signal_rel_err(actual: torch.Tensor, expected: torch.Tensor,
                   quantile: float = 0.5) -> float:
    """只在 |expected| 较大的位置上统计逐元素相对误差（避开零均值近零点）。"""
    a = actual.detach().float()
    e = expected.detach().float()
    thr = torch.quantile(e.abs().flatten(), quantile)
    mask = e.abs() >= thr
    return ((a[mask] - e[mask]).abs() / (e[mask].abs() + 1e-6)).mean().item()


def max_rel_err(actual: torch.Tensor, expected: torch.Tensor) -> float:
    a = actual.detach().float()
    e = expected.detach().float()
    return ((a - e).abs() / (e.abs() + 1e-6)).max().item()
