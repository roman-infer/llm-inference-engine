"""EP：跨 rank 专家并行输出对齐单进程 MoE；EPLB 降低热点 rank 负载。"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from cluster.ep.ep_moe import EPMoE
from cluster.ep.load_balancer import LoadBalancer
from node_backend.deepseek_infer.moe.moe import MoE, MoEConfig


def test_ep_matches_moe():
    torch.manual_seed(7)
    cfg = MoEConfig(d=64, moe_inter=48, n_routed=8, n_active=2, n_shared=2)
    moe = MoE(cfg, device="cpu", dtype=torch.float32)
    x = torch.randn(12, 64)

    ref = moe.forward_grouped(x)
    got = EPMoE(moe, ep_size=4).forward(x)
    err = ((got - ref).norm() / ref.norm()).item()
    assert err < 0.02, f"EP 与单进程 MoE 误差过大: {err}"
    print(f"[OK] EP 对齐单进程 MoE (frob err={err:.6f})")


def test_load_balancer():
    lb = LoadBalancer(n_routed=8, ep_size=4)
    plan = lb.plan([100, 80, 60, 40, 20, 10, 10, 10])
    assert plan["max_after"] <= plan["max_before"], plan
    print(f"[OK] EPLB 最大负载 {plan['max_before']:.1f} -> "
          f"{plan['max_after']:.1f}, 利用率={plan['utilization']:.3f}")
    print("     副本:", plan["replicas"])


if __name__ == "__main__":
    test_ep_matches_moe()
    test_load_balancer()
    print("EP 测试通过 ✅")
