"""调度器：P/D 速率匹配（在途上限）与全局前缀复用命中。"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from config import ServingConfig
from cluster.engine import DisaggregatedEngine


def test_rate_matching():
    scfg = ServingConfig(device="cpu", dtype="float16", max_inflight=2)
    eng = DisaggregatedEngine(scfg)
    for k in range(5):
        eng.submit(f"q{k}", [k, 2, 3, 4, 5, 6, 7, 8], 4)
    assert eng.step() == 2
    assert eng.step() == 2
    assert eng.step() == 1
    assert eng.step() == 0
    print("[OK] 速率匹配：每批最多 2 个在途 (2,2,1,0)")


def test_prefix_reuse():
    scfg = ServingConfig(device="cpu", dtype="float16", max_inflight=1)
    eng = DisaggregatedEngine(scfg)
    base = list(range(16))
    eng.submit("first", base + [100, 101, 102], 2)
    eng.step()                                   # 注册前缀

    r2 = eng.submit("second", base + [200, 201, 202], 2)
    eng.step()
    assert r2.prefix_blocks == 1, r2.prefix_blocks
    print("[OK] 全局前缀复用命中：第二个请求复用 1 个块")


if __name__ == "__main__":
    test_rate_matching()
    test_prefix_reuse()
    print("调度器测试通过 ✅")
