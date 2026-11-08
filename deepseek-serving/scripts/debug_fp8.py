"""最小调试：单块 FP8 GEMM 逐步对比。"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from node_backend.deepseek_infer.fp8 import quantize as Q

torch.manual_seed(1)
m, k, n = 32, 128, 32
x = torch.randn(m, k, device="cuda", dtype=torch.float16)
w = torch.randn(n, k, device="cuda", dtype=torch.float16) * 0.1

ref = x @ w.t()

# --- blockwise 单块 ---
xq, sa = Q.tokenwise_quant(x)
wq, sw = Q.per_tensor_quant(w)  # 单块即 tensorwise
print("sa:", sa.flatten())
print("sw:", sw)
ones = torch.ones(1, device="cuda")
partial = torch._scaled_mm(xq, wq.t(), scale_a=ones, scale_b=sw, out_dtype=torch.float32)
out = partial * sa
print("ref:\n", ref)
print("out (scale_a=1,scale_b=sw, *sa):\n", out)

# 对照 A: scale_a=sa? 不行（非singleton）
# 对照：先验证 xq*sa 与 wq*sw 还原
xdeq = xq.float() * sa
wdeq = wq.float() * sw
print("x dequant err:", (xdeq - x.float()).abs().max().item())
print("w dequant err:", (wdeq - w.float()).abs().max().item())
ref2 = xdeq.half() @ wdeq.half().t()
print("out (dequant 后 matmul):\n", ref2)
