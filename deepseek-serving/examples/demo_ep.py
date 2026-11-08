"""宽专家并行演示（虚拟 rank，loopback 模拟 all-to-all）。"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from cluster.ep.ep_moe import EPMoE
from cluster.ep.load_balancer import LoadBalancer
from node_backend.deepseek_infer.moe.moe import MoE, MoEConfig

torch.manual_seed(7)
cfg = MoEConfig(d=64, moe_inter=48, n_routed=8, n_active=2, n_shared=2)
moe = MoE(cfg, device="cpu", dtype=torch.float32)
x = torch.randn(12, 64)

ref = moe.forward_grouped(x)
ep = EPMoE(moe, ep_size=4)
got = ep.forward(x)

err = ((got - ref).norm() / ref.norm()).item()
print("EP vs 单进程 MoE 误差 :", round(err, 6))

lb = LoadBalancer(n_routed=8, ep_size=4)
plan = lb.plan([100, 80, 60, 40, 20, 10, 10, 10])
print("均衡前最大负载 :", round(plan["max_before"], 2),
      " 均衡后:", round(plan["max_after"], 2))
print("专家副本数     :", plan["replicas"])
