"""并行状态：rank / EP 组 / 专家归属 / DP 组。

专家按 rank 做均衡连续切分（owner 映射），与 DeepSeek 宽 EP 中
"专家分散到各 rank、token 经 all-to-all 找专家"的布局一致。
"""
from __future__ import annotations


class ParallelState:
    def __init__(self, n_routed: int, ep_size: int, dp_size: int = 1):
        assert ep_size >= 1
        self.n_routed = n_routed
        self.ep_size = ep_size
        self.dp_size = dp_size
        self.world = ep_size * dp_size

        self.owner: dict[int, int] = {}
        self.local_exps: dict[int, list[int]] = {r: [] for r in range(ep_size)}
        for e in range(n_routed):
            r = (e * ep_size) // n_routed
            r = min(r, ep_size - 1)
            self.owner[e] = r
            self.local_exps[r].append(e)

    def owner_of(self, expert: int) -> int:
        return self.owner[expert]

    def local_experts(self, rank: int) -> list[int]:
        return self.local_exps[rank]

    def n_local(self, rank: int) -> int:
        return len(self.local_exps[rank])
