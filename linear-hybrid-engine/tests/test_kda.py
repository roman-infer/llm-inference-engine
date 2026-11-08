# -*- coding: utf-8 -*-
"""
KDA 测试：通道门控递推与块内循环版一致，且标量门时退化为 GDN。
"""
import torch

from lhengine.layers.kda import kda_recurrent, kda_block_loop, KDABlock


def make_inputs(b=2, L=100, H=4, K=16, P=16, seed=0):
    torch.manual_seed(seed)
    K_ = torch.randn(b, L, H, K)
    V_ = torch.randn(b, L, H, P)
    Q_ = torch.randn(b, L, H, K)
    alpha = torch.rand(b, L, H, K) * 0.3 + 0.7
    beta = torch.rand(b, L, H) * 0.5 + 0.5
    return K_, V_, Q_, alpha, beta


def test_recurrent_vs_block():
    K, V, Q, alpha, beta = make_inputs(L=100)
    y1, S1 = kda_recurrent(K, V, Q, alpha, beta)
    y2, S2 = kda_block_loop(K, V, Q, alpha, beta, chunk=64)
    assert torch.allclose(y1, y2, atol=1e-5, rtol=1e-5)
    assert torch.allclose(S1, S2, atol=1e-5, rtol=1e-5)


def test_scalar_gate_reduces_to_gdn():
    """通道 α 全部相同（标量）时，KDA 应与 GDN 行为一致。"""
    K, V, Q, _, beta = make_inputs(L=64)
    scalar = torch.rand(2, 64, 4) * 0.3 + 0.7
    alpha = scalar.unsqueeze(-1).expand(-1, -1, -1, 16)
    y1, _ = kda_recurrent(K, V, Q, alpha, beta)
    y2, _ = kda_block_loop(K, V, Q, alpha, beta, chunk=64)
    assert torch.allclose(y1, y2, atol=1e-5, rtol=1e-5)


def test_block_forward_finite():
    blk = KDABlock(d_model=64, n_heads=4, d_head=16, d_key=16)
    x = torch.randn(2, 50, 64)
    y, _ = blk(x)
    assert y.shape == (2, 50, 64)
    assert torch.isfinite(y).all()
