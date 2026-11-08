# CUDA 内核（可选）

本目录是 linear-hybrid-engine 的**自研 CUDA 性能路径**。不编译它们，项目也能
完整运行（PyTorch 已提供等价的 recurrent / chunkwise 实现）。编译后，PyTorch 层
会自动检测并优先使用这些内核。

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `scan_kernel.cu` | 一阶时变递推的**并行分块扫描**（块内 Hillis-Steele + 块间 carry 传播），加速 Mamba1 |
| `ssd_kernel.cu` | Mamba2 **SSD 分块内核**（块内 decay 矩阵、对偶 MCB、块内/跨块输出、状态更新） |
| `bindings.cpp` | pybind11 绑定，编译出 `lhengine.kernels.lhengine_cuda` |

## 编译前提

1. 一块 NVIDIA GPU（建议 sm_80 及以上，如 RTX 30/40 系）；
2. 已安装与本机 PyTorch **CUDA 版本匹配**的 CUDA 工具链（`nvcc`）。
   检查：`nvcc --version`，并确认 `python -c "import torch;print(torch.version.cuda)"`；
3. 已安装 PyTorch（含 `torch.utils.cpp_extension`）。

## 编译安装

在项目根目录执行：

```bash
# Windows (PowerShell)
$env:LHENGINE_CUDA="1"; pip install -e .

# Linux / macOS
LHENGINE_CUDA=1 pip install -e .
```

编译成功后验证：

```python
import torch
from lhengine.kernels.loader import cuda_available
print(cuda_available())   # 应输出 True
```

## 与官方内核的关系

这是一个**教学/工程展示**实现，刻意保持内核结构清晰可读：

- 扫描用 float32 累加，采用经典三段式（块内并行扫描 → chunk carry 扫描 → 应用 carry）；
- SSD 单块内用动态共享内存完成 decay 矩阵、对偶矩阵与状态更新，块间由 host 循环。

它没有做官方 `mamba-ssm` / NVIDIA cuTile 那种极限融合（离散化、扫描、写回融在一个
persistent kernel 中）。若要追求极致吞吐，可进一步参考：

- NVIDIA cuTile Mamba2：<https://developer.nvidia.com/blog/accelerating-mamba-2-with-high-performance-forward-pass-kernels-in-nvidia-cuda-tile/>
- mamba-ssm：<https://pypi.org/project/mamba-ssm/>
