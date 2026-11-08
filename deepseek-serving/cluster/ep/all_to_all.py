"""all-to-all 派发 / 合并（消息传递语义，经 Transport 走）。

对齐 DeepEP：token 行先按 origin rank 分布，路由后把 token 向量发给专家 owner
rank；owner 本地 grouped SwiGLU 后，把结果发回 token 的 origin rank 加权求和。
"""
from __future__ import annotations

import torch

from node_backend.deepseek_infer.moe.grouped_gemm import dispatch, grouped_swiglu
from cluster.comm.transport import LoopbackTransport
from cluster.ep.parallel_state import ParallelState

D_TAG = "dispatch"
C_TAG = "combine"


def make_transport(ep_size: int) -> LoopbackTransport:
    t = LoopbackTransport(serialize=True)
    for r in range(ep_size):
        t.register(f"ep{r}")
    return t


def dispatch_exchange(state: ParallelState, transport: LoopbackTransport,
                      x: torch.Tensor, idx: torch.Tensor,
                      weights: torch.Tensor) -> dict[int, list]:
    n = x.shape[0]
    k = idx.shape[1]
    for i in range(n):
        origin = i % state.ep_size
        for j in range(k):
            e = int(idx[i, j])
            owner = state.owner_of(e)
            transport.send(
                f"ep{origin}", f"ep{owner}",
                {"origin": origin, "tok": i, "expert": e,
                 "w": float(weights[i, j]), "vec": x[i].clone()},
                D_TAG,
            )
    buckets = {r: [] for r in range(state.ep_size)}
    for r in range(state.ep_size):
        dst = f"ep{r}"
        while transport.pending(dst, D_TAG):
            buckets[r].append(transport.recv(dst, tag=D_TAG))
    return buckets


def local_compute(state: ParallelState, buckets: dict[int, list],
                  stores: dict[int, dict]) -> list[dict]:
    msgs: list[dict] = []
    for r, items in buckets.items():
        if not items:
            continue
        local = state.local_experts(r)
        pos = {e: k for k, e in enumerate(local)}
        xloc = torch.stack([it["vec"] for it in items], 0)
        lidx = torch.tensor([[pos[it["expert"]]] for it in items],
                            device=xloc.device)
        ones = torch.ones(len(items), 1, device=xloc.device, dtype=xloc.dtype)
        d = dispatch(xloc, lidx, ones, len(local))
        pairs_sorted = grouped_swiglu(
            xloc, d, stores[r]["w_gate"], stores[r]["w_up"], stores[r]["w_down"]
        )
        # grouped_swiglu 按专家排序返回；用 tok_sorted 散射回 item(xloc 行)顺序
        pairs = torch.empty_like(pairs_sorted)
        pairs[d.tok_sorted] = pairs_sorted
        for m, it in enumerate(items):
            msgs.append({"owner": r, "origin": it["origin"], "tok": it["tok"],
                         "w": it["w"], "out": pairs[m]})
    return msgs


def combine_exchange(state: ParallelState, transport: LoopbackTransport,
                     msgs: list[dict], n_tokens: int, d_model: int,
                     device) -> torch.Tensor:
    for m in msgs:
        transport.send(
            f"ep{m['owner']}", f"ep{m['origin']}",
            {"tok": m["tok"], "w": m["w"], "out": m["out"].clone()}, C_TAG,
        )
    out = torch.zeros(n_tokens, d_model, device=device,
                      dtype=msgs[0]["out"].dtype)
    for r in range(state.ep_size):
        dst = f"ep{r}"
        while transport.pending(dst, C_TAG):
            m = transport.recv(dst, tag=C_TAG)
            out[m["tok"]] += m["w"] * m["out"]
    return out
