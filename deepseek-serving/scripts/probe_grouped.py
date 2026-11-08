"""探测 torch._grouped_mm 可用性。"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

print("torch:", torch.__version__)
print("has _grouped_mm:", hasattr(torch, "_grouped_mm"))

if hasattr(torch, "_grouped_mm"):
    E, m, k, n = 4, 32, 64, 64
    a = torch.randn(m * E, k, device="cuda", dtype=torch.float16)
    b = torch.randn(E * n, k, device="cuda", dtype=torch.float16)
    offs = (torch.arange(E + 1, device="cuda", dtype=torch.int32) * m)
    for kw in (dict(), dict(out_dtype=torch.float16)):
        try:
            r = torch._grouped_mm(a, b, offs, offs, **kw)
            print(f"_grouped_mm OK kw={kw}: {r.shape}")
        except Exception as e:
            print(f"_grouped_mm FAIL kw={kw}: {repr(e)[:200]}")
