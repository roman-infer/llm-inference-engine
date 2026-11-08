"""扫描 outlier 强度，观察 per-tensor vs blockwise 的鲁棒性边界。"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from node_backend.deepseek_infer.fp8 import quantize as Q

DEV = "cuda"
n, k = 128, 512

for mag in (100, 1000, 10000, 100000):
    torch.manual_seed(0)
    w = torch.randn(n, k, device=DEV, dtype=torch.float16)
    w[0, 0] = mag  # 单个极端 outlier，落在 block 0

    wq_t, sw_t = Q.per_tensor_quant(w)
    deq_t = wq_t.float() * sw_t
    wq_b, sc_b = Q.blockwise_weight_quant(w, 128)
    deq_b = (wq_b.reshape(n, 4, 128).float() * sc_b.reshape(1, 4, 1)).reshape(n, k)

    # 小 block（1..3）的误差
    et = (deq_t[:, 128:] - w[:, 128:].float()).norm() / w[:, 128:].float().norm()
    eb = (deq_b[:, 128:] - w[:, 128:].float()).norm() / w[:, 128:].float().norm()
    # 小 block 中有多少值被 per-tensor 量化清零
    zeros = (wq_t[:, 128:] == 0).float().mean().item()
    print(f"outlier={mag:>7} sw_t={sw_t.item():9.2f} | small-block err tensorwise={et:.4f} "
          f"blockwise={eb:.4f} | 小block清零率={zeros:.3f}")
