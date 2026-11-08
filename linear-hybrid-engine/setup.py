# -*- coding: utf-8 -*-
"""
linear-hybrid-engine 的安装脚本。

安装方式：
    pip install -e .                  # 仅安装 PyTorch 路径（无需 nvcc，任何环境可用）
    LHENGINE_CUDA=1 pip install -e .  # 同时编译 CUDA 内核（需要本机 CUDA 工具链 + nvcc）

说明：
    项目默认提供完整的 PyTorch 实现（recurrent 递推 + chunkwise 分块并行），
    不编译任何 CUDA 代码也能完整运行全部测试与生成。
    CUDA 内核是可选的性能加速路径，源码位于 lhengine/kernels/csrc，
    编译成功后 PyTorch 层会自动优先使用（见 lhengine/kernels/loader.py）。
"""
import os
from setuptools import setup, find_packages

# 是否编译 CUDA 扩展：通过环境变量 LHENGINE_CUDA=1 开启。
WITH_CUDA = os.environ.get("LHENGINE_CUDA", "0") == "1"

ext_modules = []
if WITH_CUDA:
    # 只有在请求 CUDA 时才导入 torch 的扩展构建工具，避免无 torch/无 CUDA 环境报错。
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    csrc = os.path.join("lhengine", "kernels", "csrc")
    # 扩展模块名必须与 Python 导入路径一致：lhengine.kernels.lhengine_cuda。
    ext_modules.append(
        CUDAExtension(
            name="lhengine.kernels.lhengine_cuda",
            sources=[
                os.path.join(csrc, "scan_kernel.cu"),
                os.path.join(csrc, "ssd_kernel.cu"),
                os.path.join(csrc, "bindings.cpp"),
            ],
            extra_compile_args={
                # nvcc 选项：现代架构（Ampere/Ada），保留行号信息。
                "nvcc": ["-O3", "-lineinfo"],
                "cxx": ["-O3"],
            },
        )
    )

# 自定义 cmdclass：仅在编译 CUDA 时使用 BuildExtension。
cmdclass = {}
if WITH_CUDA:
    from torch.utils.cpp_extension import BuildExtension
    cmdclass["build_ext"] = BuildExtension

setup(
    name="linear-hybrid-engine",
    version="0.1.0",
    description="后 Transformer 线性/混合架构推理引擎：Mamba1/2(SSD)、DeltaNet/GatedDeltaNet、KDA、Hybrid",
    packages=find_packages(exclude=("tests", "examples", "docs")),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.1",
        "safetensors>=0.4",
        "tokenizers>=0.15",
        "huggingface_hub>=0.20",
        "numpy>=1.24",
    ],
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
