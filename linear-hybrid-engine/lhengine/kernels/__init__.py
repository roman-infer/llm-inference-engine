# -*- coding: utf-8 -*-
"""
kernels 子包：序列扫描（scan）原语。

scan 是线性/SSM 架构的核心算子：把一个一阶递推
    h_t = a_t · h_{t-1} + b_t
在训练/prefill 阶段用并行（分块）方式算出来，而不是逐步循环。

本项目提供：
    - naive_scan  : 朴素循环（参考实现，用来与并行版做数值对齐）；
    - block_scan : 分块并行 scan（块内循环、块间并行，GPU 友好）；
    - scan       : 默认入口。

另外 csrc/ 目录下提供等价的 CUDA 实现（可选编译，见 csrc/README），
在装好 CUDA 工具链后可通过 setup.py 编译，替换这里的纯 PyTorch 版本。
"""
from .scan import naive_scan, block_scan, scan
from .loader import cuda_ext, cuda_available

__all__ = ["naive_scan", "block_scan", "scan",
           "cuda_ext", "cuda_available"]
