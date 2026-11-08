# -*- coding: utf-8 -*-
"""
Gated DeltaNet 测试：验证 delta rule 的递推与 WY/UT chunkwise 等价。
"""
import torch
import torch.nn.functional as F

from lhengine.layers.deltanet import (delta_recurrent, delta_chunkwise,
                                     GatedDeltaNetBlock)


def make_inputs(b=2, L=130, H=4, K=16, P=16, gated=True, seed=0):
    """构造随机的 delta 输入；alpha 为遗忘门（取 0.5~1 避免过快衰减）。

    与真实 GDN 一致，对 key/query 做 L2 归一化，否则 delta 状态会因 ||k||>1
    而指数发散，chunkwise 的线性求解也会奇异。
    """
    torch.manual_seed(seed)
    K_ = F.normalize(torch.randn(b, L, H, K), dim=-1)
    V_ = torch.randn(b, L, H, P)
    Q_ = F.normalize(torch.randn(b, L, H, K), dim=-1)
    beta = torch.rand(b, L, H) * 0.5 + 0.5
    if gated:
        alpha = torch.rand(b, L, H) * 0.3 + 0.7
    else:
        alpha = torch.ones(b, L, H)
    return K_, V_, Q_, alpha, beta


def test_recurrent_vs_chunkwise_gated():
    """gated delta：循环版与 chunkwise 应一致。"""
    K, V, Q, alpha, beta = make_inputs(L=130, gated=True)
    y_rec, S_rec = delta_recurrent(K, V, Q, alpha, beta)
    y_chk, S_chk = delta_chunkwise(K, V, Q, alpha, beta, chunk=64)
    assert torch.allclose(y_rec, y_chk, atol=1e-4, rtol=1e-4)
    assert torch.allclose(S_rec, S_chk, atol=1e-4, rtol=1e-4)


def test_recurrent_vs_chunkwise_ungated():
    """无门控 delta：循环版与 chunkwise 应一致。"""
    K, V, Q, alpha, beta = make_inputs(L=130, gated=False)
    y_rec, S_rec = delta_recurrent(K, V, Q, alpha, beta)
    y_chk, S_chk = delta_chunkwise(K, V, Q, alpha, beta, chunk=64)
    assert torch.allclose(y_rec, y_chk, atol=1e-4, rtol=1e-4)
    assert torch.allclose(S_rec, S_chk, atol=1e-4, rtol=1e-4)


def test_block_modes():
    """GatedDeltaNetBlock 的 recurrent / chunkwise 输出应一致。"""
    blk = GatedDeltaNetBlock(d_model=64, n_heads=4, d_head=16, d_key=16, gated=True)
    x = torch.randn(2, 100, 64)
    y1, _ = blk(x, mode="recurrent")
    y2, _ = blk(x, mode="chunkwise")
    assert torch.allclose(y1, y2, atol=1e-4, rtol=1e-4)
