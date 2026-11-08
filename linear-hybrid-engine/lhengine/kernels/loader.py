# -*- coding: utf-8 -*-
"""
CUDA 扩展加载器：尝试导入编译好的 lhengine_cuda，失败则回退到纯 PyTorch。

设计原则：
    - 不编译 CUDA（绝大多数环境）也能完整运行，本模块返回 None；
    - 编译并安装后，cuda_ext() 返回扩展模块，调用方优先使用其内核。
这样"性能路径"是可选的，项目在 CPU / 无 nvcc 环境下依旧可用。
"""
import torch

_CUDA_EXT = None
_TRIED = False


def cuda_ext():
    """返回编译好的 CUDA 扩展模块；不可用时返回 None。"""
    global _CUDA_EXT, _TRIED
    if _TRIED:
        return _CUDA_EXT
    _TRIED = True
    # 必须同时满足：在 CUDA 设备上、且扩展已编译。
    if torch.cuda.is_available():
        try:
            from . import lhengine_cuda  # 由 setup.py 编译生成
            _CUDA_EXT = lhengine_cuda
        except Exception:
            _CUDA_EXT = None
    return _CUDA_EXT


def cuda_available() -> bool:
    """是否可使用自研 CUDA 内核。"""
    return cuda_ext() is not None
