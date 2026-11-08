"""Decode 引擎：接收 KV 块写入本地 cache，导入首个 token，逐 token 解码。"""
from __future__ import annotations

import torch

from cluster.comm.kv_connector import KVConnector
from cluster.comm.transport import LoopbackTransport
from node_backend.deepseek_infer.engine.sampling import greedy
from cluster.pd.node_factory import build_node


class DecodeEngine:
    def __init__(self, eid: str, scfg, mcfg, block_size: int = 16):
        self.eid = eid
        self.B = block_size
        dtype = {"float16": torch.float16,
                 "float32": torch.float32}[scfg.dtype]
        self.model, self.cache = build_node(
            mcfg, scfg.seed, scfg.device, dtype)

    def serve(self, req, src: str, transport: LoopbackTransport,
              connector: KVConnector) -> list[int]:
        ctrl = transport.recv(self.eid, src, "pd_ctrl")
        blocks = connector.recv(transport, src, self.eid)

        self.cache.reset()
        for bd in blocks:
            st = bd.logical_index * self.B
            self.cache.ckv[:, st:st + bd.fill].copy_(bd.ckv)
            self.cache.krope[:, st:st + bd.fill].copy_(bd.krope)

        S = ctrl["S"]
        nxt = ctrl["first"]
        generated = [nxt]
        cur = S
        for _ in range(req.max_new - 1):
            logits, _ = self.model.forward_decode(
                torch.tensor(nxt, device=req.prompt.device), cur)
            nxt = int(greedy(logits[0, -1]).item())
            generated.append(nxt)
            cur += 1
        return generated
