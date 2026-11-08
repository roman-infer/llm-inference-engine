"""通用工具：数值比对、CUDA 计时、显存探测。"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Callable

import torch


# FP8 E4M3 可表示的最大有限值
FP8_E4M3_MAX = 448.0


def assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor,
                 atol: float = 1e-2, rtol: float = 1e-2) -> float:
    """数值比对，返回最大绝对误差。"""
    actual = actual.detach().to(torch.float32).cpu()
    expected = expected.detach().to(torch.float32).cpu()
    diff = (actual - expected).abs()
    max_err = diff.max().item()
    tol = atol + rtol * expected.abs()
    if not (diff <= tol).all():
        idx = (diff > tol).nonzero()[0]
        raise AssertionError(
            f"[{name}] 数值不一致 max_err={max_err:.4f} atol={atol} rtol={rtol}; "
            f"首个失败位置={idx.tolist()} actual={actual[tuple(idx)]:.4f} "
            f"expected={expected[tuple(idx)]:.4f}"
        )
    return max_err


@dataclass
class BenchResult:
    name: str
    mean_ms: float
    median_ms: float
    min_ms: float
    iters: int

    def __repr__(self) -> str:
        return (f"{self.name}: mean={self.mean_ms:.4f}ms median={self.median_ms:.4f}ms "
                f"min={self.min_ms:.4f}ms (iters={self.iters})")


def benchmark_cuda(fn: Callable[[], object], warmup: int = 10, iters: int = 50,
                    name: str = "") -> BenchResult:
    """用 CUDA event 精确计时；自动处理需要同步的函数。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    return BenchResult(
        name=name or getattr(fn, "__name__", "fn"),
        mean_ms=statistics.mean(times),
        median_ms=statistics.median(times),
        min_ms=min(times),
        iters=iters,
    )


def free_vram_gb() -> float:
    free, total = torch.cuda.mem_get_info()
    return free / 1e9


def sync_time_s(fn: Callable[[], object]) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t0
