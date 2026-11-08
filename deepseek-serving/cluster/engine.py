"""顶层编排：组装 Transport / KV 池 / 分层存储 / Connector / P·D 引擎 / 调度器。

在单进程内用 LoopbackTransport 模拟一个"P 集群 + D 集群"，提供：
- submit / run：提交并跑完全部请求；
- reference：同 seed 的节点后端 Generator 单机基线，用于校验 PD 贪心输出无损。
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

from config import ServingConfig
from node_backend.deepseek_infer.engine import Generator
from node_backend.deepseek_infer.model import mini_config
from cluster.comm.kv_connector import KVConnector
from cluster.comm.transport import LoopbackTransport
from cluster.kvpool.block_pool import BlockPool
from cluster.kvpool.prefix_index import PrefixIndex
from cluster.kvpool.tiered_store import TieredStore
from cluster.pd.decode_engine import DecodeEngine
from cluster.pd.prefill_engine import PrefillEngine
from cluster.scheduler.global_scheduler import GlobalScheduler
from cluster.scheduler.request import Request

_DT = {"float16": torch.float16, "float32": torch.float32}


class DisaggregatedEngine:
    def __init__(self, scfg: ServingConfig | None = None, mcfg=None):
        self.s = scfg or ServingConfig()
        self.mcfg = mcfg or mini_config()
        self.dtype = _DT[self.s.dtype]

        self.transport = LoopbackTransport(serialize=True)
        self.transport.register("P")
        self.transport.register("D")

        self.pool = BlockPool(self.mcfg, self.s.device, self.dtype)
        self.store = TieredStore(
            self.s.gpu_cap, self.s.cpu_cap, self.s.ssd_cap,
            self.s.ssd_dir, self.s.device)
        self.connector = KVConnector()

        self.P = PrefillEngine("P", self.s, self.mcfg)
        self.D = DecodeEngine("D", self.s, self.mcfg, self.pool.B)

        self.pindex = PrefixIndex(self.pool.B)
        self.scheduler = GlobalScheduler(
            ["P"], ["D"], self.s.max_inflight, self.pindex)

    # ---------- 请求 ----------
    def submit(self, rid: str, prompt, max_new: int,
               arrival: int = 0) -> Request:
        if not torch.is_tensor(prompt):
            prompt = torch.tensor(prompt)
        req = Request(rid, prompt.to(self.s.device), max_new, arrival)
        self.scheduler.submit(req)
        return req

    def step(self) -> int:
        admitted = self.scheduler.admit()
        for req, pe, de in admitted:
            self.P.prefill(req, de, self.transport, self.connector, self.pool)
            req.output = self.D.serve(req, pe, self.transport, self.connector)
            stub = SimpleNamespace(seq_id=de)
            self.pindex.register(req.prompt, stub)
            self.scheduler.finish(req)
        self.store.enforce_caps(self.pool)
        return len(admitted)

    def run(self) -> dict[str, list[int]]:
        while self.step():
            pass
        return {r.rid: r.output for r in self.scheduler.done}

    def full_text(self, req: Request) -> torch.Tensor:
        return torch.cat([req.prompt.cpu(), torch.tensor(req.output)])

    # ---------- 单机基线（无损校验）----------
    def reference(self, req: Request) -> torch.Tensor:
        from node_backend.deepseek_infer.model import DeepSeekModel
        from node_backend.deepseek_infer.engine.kv_cache import LatentKVCache
        torch.manual_seed(self.s.seed)
        m = DeepSeekModel(self.mcfg, device=self.s.device, dtype=self.dtype)
        c = LatentKVCache(self.mcfg, device=self.s.device, dtype=self.dtype)
        g = Generator(m, None, c, self.s.device)
        return g.generate(req.prompt, req.max_new)
