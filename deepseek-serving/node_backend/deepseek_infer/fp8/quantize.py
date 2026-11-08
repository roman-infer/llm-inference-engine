"""FP8 量化：per-tensor / per-token(rowwise) / 1x128 block-wise（DeepSeek 方案）。

DeepSeek-V3 权重量化口径：输入通道(K)方向每 128 个元素共享一个 scale，
输出通道(N)方向共享，即 block shape = 1x128；激活为 per-token(rowwise) 量化。
"""
from __future__ import annotations

import torch

from ..utils import FP8_E4M3_MAX


def _amax(x: torch.Tensor, dim=None, keepdim=False) -> torch.Tensor:
    return x.detach().abs().amax(dim=dim, keepdim=keepdim).clamp_min(1e-12)


def make_scale(amax: torch.Tensor) -> torch.Tensor:
    return (amax / FP8_E4M3_MAX).to(torch.float32)


def cast_to_fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """按 scale 缩放后裁剪到 E4M3 范围并转为 fp8。"""
    scaled = (x / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    return scaled.to(torch.float8_e4m3fn)


def per_tensor_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """整张量一个 scale。返回 (x_fp8, scale标量张量)。"""
    scale = make_scale(_amax(x))
    return cast_to_fp8(x, scale), scale


def tokenwise_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """激活 per-token 量化：x shape (M, K) -> scale shape (M, 1)。"""
    scale = make_scale(_amax(x, dim=-1, keepdim=True))
    return cast_to_fp8(x, scale), scale


def blockwise_weight_quant(w: torch.Tensor, block_size: int = 128
                           ) -> tuple[torch.Tensor, torch.Tensor]:
    """权重 1x128 block 量化。

    w: (N, K)；K 必须能被 block_size 整除。
    返回 w_fp8 (N, K) 与 scales (K/block_size,)，每个 K-block 一个 scale（N 方向共享）。
    """
    n, k = w.shape
    assert k % block_size == 0, f"K={k} 不能被 block_size={block_size} 整除"
    w_blocks = w.reshape(n, k // block_size, block_size)
    scales = make_scale(_amax(w_blocks, dim=(0, 2)))  # (K/block_size,)
    wq_blocks = cast_to_fp8(w_blocks, scales.reshape(1, -1, 1))
    return wq_blocks.reshape(n, k), scales


def dequantize(x_fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x_fp8.to(torch.float32) * scale


def pad_k(x: torch.Tensor, block_size: int) -> tuple[torch.Tensor, int]:
    """K 维右侧 pad 到 block_size 的整数倍（pad 0，不影响 GEMM 结果）。"""
    k = x.shape[-1]
    pad = (block_size - k % block_size) % block_size
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    return x, k




