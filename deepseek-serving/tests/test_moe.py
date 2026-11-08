"""MoE 数值正确性：路由、dispatch、grouped vs naive、共享专家。"""
import torch

from common import frobenius_rel_err
from node_backend.deepseek_infer.moe import MoE, MoEConfig
from node_backend.deepseek_infer.moe.grouped_gemm import dispatch

torch.manual_seed(7)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float16
CFG = MoEConfig(d=64, moe_inter=32, n_routed=8, n_active=2, n_shared=2)


def test_routing():
    moe = MoE(CFG, device=DEV, dtype=DT)
    x = torch.randn(48, CFG.d, device=DEV, dtype=DT)
    idx, weights = moe._route(x)
    assert idx.shape == (48, 2) and weights.shape == (48, 2)
    assert torch.allclose(weights.sum(-1).float(), torch.ones(48, device=DEV), atol=1e-2)
    assert idx.min() >= 0 and idx.max() < CFG.n_routed
    print("[OK] top-k 路由 shape、权重归一化与索引范围正确")


def test_dispatch_counts():
    moe = MoE(CFG, device=DEV, dtype=DT)
    x = torch.randn(48, CFG.d, device=DEV, dtype=DT)
    idx, weights = moe._route(x)
    d = dispatch(x, idx, weights, CFG.n_routed)
    assert d.counts.sum().item() == 48 * 2
    # mask 有效数 == counts
    assert d.mask.sum().item() == d.counts.sum().item()
    print(f"[OK] dispatch 分组正确，专家负载: {d.counts.tolist()}")


def test_grouped_vs_naive():
    moe = MoE(CFG, device=DEV, dtype=DT)
    x = torch.randn(48, CFG.d, device=DEV, dtype=DT)
    og = moe.forward_grouped(x)
    on = moe.forward_naive(x)
    err = frobenius_rel_err(og, on)
    print(f"[info] grouped vs naive Frobenius err={err:.4f}")
    assert err < 0.05, f"grouped 与 naive 不一致: {err}"
    print("[OK] padded bmm grouped 路径与显式 naive 路径数值一致")


def test_shared_experts():
    moe = MoE(CFG, device=DEV, dtype=DT)
    x = torch.randn(16, CFG.d, device=DEV, dtype=DT)
    with torch.no_grad():
        moe.se_gate.fill_(1.0)
    full = moe.forward_grouped(x)
    with torch.no_grad():
        moe.se_gate.fill_(0.0)
    routed_only = moe.forward_grouped(x)
    assert not torch.allclose(full, routed_only, atol=1e-3), "共享专家未生效"
    print("[OK] 共享专家及其 gate 标量正确参与输出")


if __name__ == "__main__":
    test_routing()
    test_dispatch_counts()
    test_grouped_vs_naive()
    test_shared_experts()
    print("\n全部 MoE 测试通过 ✅")
