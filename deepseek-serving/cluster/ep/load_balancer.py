"""EPLB-lite：专家负载均衡。

热点专家增加副本、把副本放到最空闲的 rank；专家有 k 个副本时其负载由 k 个副本
均摊。迭代直到最大 rank 负载不再明显下降。对齐 DeepSeek EPLB 的思路（提升专家
利用率、消除瓶颈 rank）。
"""
from __future__ import annotations


class LoadBalancer:
    def __init__(self, n_routed: int, ep_size: int, max_replicas: int = 4):
        self.E = n_routed
        self.R = ep_size
        self.max_replicas = max_replicas

    def plan(self, loads):
        loads = [float(x) for x in loads]
        # home rank（连续切分），每专家先 1 副本
        reps = {e: 1 for e in range(self.E)}
        placement = {r: [] for r in range(self.R)}
        for e in range(self.E):
            r = min((e * self.R) // self.E, self.R - 1)
            placement[r].append(e)

        def rank_loads():
            rl = [0.0] * self.R
            for r, exps in placement.items():
                for e in exps:
                    rl[r] += loads[e] / reps[e]
            return rl

        before = rank_loads()
        base_max = max(before)

        while True:
            rl = rank_loads()
            cur_max = max(rl)
            # 有效负载最高的专家（副本最少收益最大）
            e_star = max(range(self.E), key=lambda e: loads[e] / reps[e])
            if reps[e_star] >= self.max_replicas:
                break
            r_star = min(range(self.R), key=lambda r: rl[r])
            # 试放置
            placement[r_star].append(e_star)
            reps[e_star] += 1
            new_max = max(rank_loads())
            if new_max >= cur_max * 0.999:           # 不再改善 -> 回滚
                placement[r_star].pop()
                reps[e_star] -= 1
                break

        after = rank_loads()
        mean = sum(after) / self.R
        return {
            "replicas": reps,
            "placement": placement,
            "max_before": base_max,
            "max_after": max(after),
            "utilization": (max(after) / mean) if mean > 0 else 0.0,
        }
