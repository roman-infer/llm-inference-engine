"""复现 k=512 四块循环的误差问题。"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from node_backend.deepseek_infer.fp8 import FP8Linear

torch.manual_seed(0)
m, k, n = 64, 512, 128
lin = FP8Linear(k, n, block_size=128, device="cuda")
x = torch.randn(m, k, device="cuda", dtype=torch.float16)
ref = lin.forward_fp16(x)
out_b = lin.forward_blockwise(x)


def relerr(a, b):
    return ((a.float() - b.float()).abs() / (b.float().abs() + 1e-6)).mean().item()


print("blockwise rel err:", relerr(out_b, ref))

# 手动 dequant 路径：把 fp8 权重按块反量化、fp16 matmul
w = lin.wq_blockwise.float()
for j in range(4):
    w[:, j * 128:(j + 1) * 128] *= lin.block_scales[j]
xq, sa = (lambda t: (t[0], t[1]))(__import__("deepseek_infer.fp8.quantize", fromlist=["x"]).tokenwise_quant(x))
xdeq = (xq.float() * sa).half()
out_manual = xdeq @ w.half().t()
print("dequant 后 matmul rel err:", relerr(out_manual, ref))

# 逐块 partial 检查
xq2, sa2 = __import__("deepseek_infer.fp8.quantize", fromlist=["x"]).tokenwise_quant(x)
acc = torch.zeros(m, n, device="cuda")
ones = torch.ones(1, device="cuda")
for j in range(4):
    xb = xq2[:, j * 128:(j + 1) * 128].contiguous()
    wb = lin.wq_blockwise[:, j * 128:(j + 1) * 128].contiguous()
    p = torch._scaled_mm(xb, wb.t(), scale_a=ones, scale_b=lin.block_scales[j].reshape(1),
                         out_dtype=torch.float32)
    refp = x[:, j * 128:(j + 1) * 128].float() @ wb.float().t() * lin.block_scales[j]
    print(f"block {j}: partial norm={p.norm():.3f} naive norm={refp.norm():.3f} "
          f"relerr={relerr(p, refp):.4f} sw={lin.block_scales[j].item():.6f}")
    acc += p
out2 = (acc * sa2).half()
print("reconstructed rel err:", relerr(out2, ref))
