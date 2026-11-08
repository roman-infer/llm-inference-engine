"""PD 分离演示（单进程 loopback 模拟 P/D 集群）。"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from config import ServingConfig
from cluster.engine import DisaggregatedEngine

scfg = ServingConfig(device="cpu", dtype="float16", seed=123)
eng = DisaggregatedEngine(scfg)

prompt = [1, 20, 300, 5, 90, 7, 8, 33, 4, 9]
req = eng.submit("demo", prompt, 16)
eng.run()

full = eng.full_text(req)
ref = eng.reference(req)

print("PD 生成 token :", req.output)
print("PD 是否无损   :", bool((full == ref).all().item()))
print("KV 分层块数   :", eng.store.tier_counts(eng.pool))
print("传输字节数    :", eng.transport.bytes_sent)
