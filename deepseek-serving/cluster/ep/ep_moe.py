"""EP MoE：把节点后端单进程 MoE 重写为跨 rank 专家并行，数值对齐单进程基线。"""
from __future__ import annotations

import torch

from node_backend.deepseek_infer.moe.moe import MoE
from cluster.comm.transport import LoopbackTransport
from cluster.ep import all_to_all
from cluster.ep.parallel_state import ParallelState


def build_local_stores(moe: MoE, state: ParallelState) -> dict[int, dict]:
    stores: dict[int, dict] = {}
    for r in range(state.ep_size):
        exps = state.local_experts(r)
        idx = torch.tensor(exps, device=moe.w_gate.device)
        stores[r] = {
            "w_gate": moe.w_gate[idx].clone(),
            "w_up": moe.w_up[idx].clone(),
            "w_down": moe.w_down[idx].clone(),
        }
    return stores


class EPMoE:
    def __init__(self, moe: MoE, ep_size: int, dp_size: int = 1,
                 transport: LoopbackTransport | None = None):
        self.ref = moe
        self.state = ParallelState(moe.cfg.n_routed, ep_size, dp_size)
        self.stores = build_local_stores(moe, self.state)
        self.transport = transport or all_to_all.make_transport(ep_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx, weights = self.ref._route(x)
        buckets = all_to_all.dispatch_exchange(
            self.state, self.transport, x, idx, weights)
        msgs = all_to_all.local_compute(self.state, buckets, self.stores)
        routed = all_to_all.combine_exchange(
            self.state, self.transport, msgs, x.shape[0],
            self.ref.cfg.d, x.device)
        return routed + self.ref._shared(x)
