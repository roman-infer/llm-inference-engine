# -*- coding: utf-8 -*-
"""
Mamba2 SSD 测试：验证"递推视图"与"chunkwise 矩阵视图"严格等价。

这是整个项目数学正确性的核心：两种算法实现必须给出相同结果。
"""
import torch

from lhengine.layers.mamba2 import ssd_recurrent, ssd_chunkwise, Mamba2Block


def make_inputs(b=2, L=130, H=4, P=16, N=32, seed=0):
    """构造随机的 SSD 输入（小尺寸，便于精确比对）。"""
    torch.manual_seed(seed)
    X = torch.randn(b, L, H, P)
    # log_dA 取负值，使衰减在 (0,1)
    log_dA = -torch.rand(b, L, H, N).abs()
    B = torch.randn(b, L, H, N)
    C = torch.randn(b, L, H, N)
    return X, log_dA, B, C


def test_recurrent_vs_chunkwise():
    """SSD 循环版与分块版结果应一致（含非整块长度）。"""
    X, log_dA, B, C = make_inputs(L=130)  # 130 不是 64 的整数倍，验证边界
    y_rec, S_rec = ssd_recurrent(X, log_dA.exp(), B, C)
    y_chk, S_chk = ssd_chunkwise(X, log_dA, B, C, chunk=64)
    assert torch.allclose(y_rec, y_chk, atol=1e-4, rtol=1e-4)
    assert torch.allclose(S_rec, S_chk, atol=1e-4, rtol=1e-4)


def test_chunk_sizes():
    """不同 chunk 大小都应与循环版一致。"""
    X, log_dA, B, C = make_inputs(L=200)
    y_ref, _ = ssd_recurrent(X, log_dA.exp(), B, C)
    for ch in (16, 32, 64, 128):
        y, _ = ssd_chunkwise(X, log_dA, B, C, chunk=ch)
        assert torch.allclose(y_ref, y, atol=1e-4, rtol=1e-4), f"chunk={ch} 不一致"


def test_initial_state():
    """带初始状态时两版仍应一致。"""
    X, log_dA, B, C = make_inputs(L=100)
    S0 = torch.randn(2, 4, 32, 16)
    y_rec, S_rec = ssd_recurrent(X, log_dA.exp(), B, C, S0=S0)
    y_chk, S_chk = ssd_chunkwise(X, log_dA, B, C, chunk=64, S0=S0)
    assert torch.allclose(y_rec, y_chk, atol=1e-4, rtol=1e-4)
    assert torch.allclose(S_rec, S_chk, atol=1e-4, rtol=1e-4)


def test_block_forward_modes():
    """Mamba2Block 的 recurrent / chunkwise 两种模式输出应一致。"""
    blk = Mamba2Block(d_model=64, d_inner=128, n_heads=4, d_state=32, chunk=64)
    x = torch.randn(2, 100, 64)
    y1, _ = blk(x, mode="recurrent")
    y2, _ = blk(x, mode="chunkwise")
    assert torch.allclose(y1, y2, atol=1e-4, rtol=1e-4)
