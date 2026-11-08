"""全局 KV 调度器：准入、SLO 感知的 P/D 放置、P/D 速率匹配、前缀感知路由。"""
from __future__ import annotations

from cluster.kvpool.prefix_index import PrefixIndex
from cluster.scheduler.request import Request


class GlobalScheduler:
    def __init__(self, p_ids: list[str], d_ids: list[str],
                 max_inflight: int, prefix_index: PrefixIndex):
        self.p_ids = p_ids
        self.d_ids = d_ids
        self.max_inflight = max_inflight
        self.pindex = prefix_index
        self.queued: list[Request] = []
        self.active: list[Request] = []
        self.done: list[Request] = []
        self._placed = 0

    def submit(self, req: Request) -> None:
        self.queued.append(req)

    def admit(self) -> list[tuple[Request, str, str]]:
        admitted = []
        while self.queued and len(self.active) < self.max_inflight:
            req = self.queued.pop(0)
            pe = self.p_ids[self._placed % len(self.p_ids)]
            de = self.d_ids[self._placed % len(self.d_ids)]
            self._placed += 1
            hit = self.pindex.match(req.prompt)
            if hit is not None:
                req.prefix_blocks = hit[1]
            req.status = "prefilling"
            self.active.append(req)
            admitted.append((req, pe, de))
        return admitted

    def finish(self, req: Request) -> None:
        req.status = "done"
        if req in self.active:
            self.active.remove(req)
        self.done.append(req)
