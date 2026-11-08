"""PD 端到端：解耦式 PD 贪心输出必须与单机基线逐 token 一致（无损）。"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from config import ServingConfig
from cluster.engine import DisaggregatedEngine


def test_pd_lossless():
    scfg = ServingConfig(device="cpu", dtype="float16", seed=123)
    eng = DisaggregatedEngine(scfg)
    prompt = [1, 20, 300, 5, 90, 7, 8, 33, 4, 9]
    req = eng.submit("r1", prompt, 20)
    eng.run()

    full = eng.full_text(req)
    ref = eng.reference(req)
    assert full.shape == ref.shape, (full.shape, ref.shape)
    mismatch = (full != ref).nonzero()
    if len(mismatch):
        i = mismatch[0].item()
        raise AssertionError(
            f"PD 与单机基线在位置 {i} 不一致: pd={full[i].item()} "
            f"ref={ref[i].item()}")
    print("[OK] PD 解耦输出与单机基线逐 token 一致（贪心无损）")
    print("     生成:", req.output)


if __name__ == "__main__":
    test_pd_lossless()
    print("PD 端到端测试通过 ✅")
