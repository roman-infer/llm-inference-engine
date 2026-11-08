"""Prefill 引擎：分块（chunked）prefill，把 latent KV 切块入池并经 Connector 发往 D。"""
from __future__ import annotations

import torch

from cluster.comm.kv_connector import KVConnector
from cluster.comm.transport import LoopbackTransport
from cluster.kvpool.block_pool import BlockPool, SequenceKVMeta
from node_backend.deepseek_infer.engine.sampling import greedy
from cluster.pd.node_factory import build_node


class PrefillEngine:
    def __init__(self, eid: str, scfg, mcfg, chunk: int = 128):
        self.eid = eid
        self.chunk = chunk
        dtype = {"float16": torch.float16,
                 "float32": torch.float32}[scfg.dtype]
        self.model, self.cache = build_node(
            mcfg, scfg.seed, scfg.device, dtype)

    def prefill(self, req, dst: str, transport: LoopbackTransport,
                connector: KVConnector, pool: BlockPool):
        self.cache.reset()
        p = req.prompt
        S = p.shape[0]
        logits = None
        for s0 in range(0, S, self.chunk):
            part = p[s0:s0 + self.chunk]
            if s0 == 0:
                logits, _ = self.model.forward_prefill(part)
            else:
                logits, _ = self.model.forward_continue(part, s0)

        first = int(greedy(logits[0, -1]).item())

        meta = SequenceKVMeta(req.rid, pool)
        pool.append_tokens(meta, self.cache.ckv[:, :S], self.cache.krope[:, :S])
        blocks = [pool.to_block_data(meta, j) for j in sorted(meta.table)]

        connector.send(transport, self.eid, dst, blocks)
        transport.send(self.eid, dst,
                       {"S": S, "first": first}, "pd_ctrl")
        return meta, first
