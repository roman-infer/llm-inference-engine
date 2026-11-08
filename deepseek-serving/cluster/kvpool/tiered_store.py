"""分层 KV 存储：GPU -> CPU DRAM -> SSD，卸载/加载与容量淘汰（LRU）。"""
from __future__ import annotations

import os
import tempfile

import torch

from cluster.kvpool.block_pool import BlockPool


class TieredStore:
    def __init__(self, gpu_cap=0, cpu_cap=0, ssd_cap=0, ssd_dir: str = "",
                 gpu_device="cpu"):
        self.gpu_device = gpu_device
        self.cpu_device = torch.device("cpu")
        self.ssd_dir = ssd_dir or tempfile.mkdtemp(prefix="kvssd_")
        os.makedirs(self.ssd_dir, exist_ok=True)
        self.caps = {"gpu": gpu_cap, "cpu": cpu_cap, "ssd": ssd_cap}

    def _path(self, bid: int) -> str:
        return os.path.join(self.ssd_dir, f"block_{bid}.pt")

    def move(self, pool: BlockPool, bid: int, to: str) -> None:
        b = pool.blocks[bid]
        frm = b.tier
        if frm == to:
            return
        if to == "ssd":
            torch.save({"ckv": b.ckv, "krope": b.krope}, self._path(bid))
            b.ckv = None
            b.krope = None
        elif frm == "ssd":
            d = torch.load(self._path(bid), weights_only=False)
            dev = self.gpu_device if to == "gpu" else self.cpu_device
            b.ckv = d["ckv"].to(dev)
            b.krope = d["krope"].to(dev)
        else:
            dev = self.gpu_device if to == "gpu" else self.cpu_device
            b.ckv = b.ckv.to(dev)
            b.krope = b.krope.to(dev)
        b.tier = to

    def _on_tier(self, pool: BlockPool, tier: str) -> list[int]:
        return [bid for bid, b in pool.blocks.items() if b.tier == tier]

    def enforce_caps(self, pool: BlockPool) -> int:
        moves = 0
        for frm, to in (("gpu", "cpu"), ("cpu", "ssd")):
            cap = self.caps[frm]
            if cap <= 0:
                continue
            while len(self._on_tier(pool, frm)) > cap:
                victims = sorted(
                    ((bid, b) for bid, b in pool.blocks.items()
                     if b.tier == frm and b.refs == 0),
                    key=lambda x: x[1].last_used,
                )
                if not victims:
                    break
                self.move(pool, victims[0][0], to)
                moves += 1
        return moves

    def tier_counts(self, pool: BlockPool) -> dict[str, int]:
        return {t: len(self._on_tier(pool, t)) for t in ("gpu", "cpu", "ssd")}
