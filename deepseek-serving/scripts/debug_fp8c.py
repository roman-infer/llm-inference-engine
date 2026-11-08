"""检查反量化是否真的还原了权重和激活。"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from node_backend.deepseek_infer.fp8 import FP8Linear
from node_backend.deepseek_infer.fp8 import quantize as Q

torch.manual_seed(0)
m, k, n = 64, 512, 128
lin = FP8Linear(k, n, block_size=128, device="cuda")
x = torch.randn(m, k, device="cuda", dtype=torch.float16)

# 权重反量化
wd = lin.wq_blockwise.float().reshape(n, 4, 128) * lin.block_scales.reshape(1, 4, 1)
wd = wd.reshape(n, k)
werr = (wd - lin.weight.float()).abs() / (lin.weight.float().abs() + 1e-6)
print("weight dequant mean rel err:", werr.mean().item(), "max:", werr.max().item())

# 激活反量化
xq, sa = Q.tokenwise_quant(x)
xd = xq.float() * sa
xerr = (xd - x.float()).abs() / (x.float().abs() + 1e-6)
print("act dequant mean rel err:", xerr.mean().item(), "max:", xerr.max().item())

# fp32 路径
ref32 = x.float() @ lin.weight.float().t()
out32 = xd @ wd.t()
r32 = ((out32 - ref32).abs() / (ref32.abs() + 1e-6)).mean().item()
print("fp32 matmul rel err:", r32)

# half 路径
outh = xd.half() @ wd.half().t()
refh = x @ lin.weight.t()
rh = ((outh.float() - refh.float()).abs() / (refh.float().abs() + 1e-6)).mean().item()
print("half matmul rel err:", rh)

# 检查 sa 量级与 x@w^T 量级
print("sa range:", sa.min().item(), sa.max().item())
print("ref norm:", refh.float().norm().item(), "out norm:", outh.float().norm().item())
