# linear-hybrid-engine · 后 Transformer 线性/混合架构推理引擎

从零实现的**后 Transformer（Post-Transformer）线性 / 混合架构推理引擎**：覆盖
Mamba1 选择性 SSM、Mamba2 SSD（recurrent + chunkwise 对偶）、DeltaNet /
Gated DeltaNet（delta rule + WY/UT 分块）、KDA（Kimi Delta Attention，逐通道门控）、
滑动窗口注意力（SWA），并把它们组装成可加载真实权重、可生成的混合语言模型。

> 核心心智模型：Transformer 把历史存成**随长度增长的 KV**；线性/混合架构把历史
> 压缩进**固定大小的状态矩阵**——decode 显存从 O(N) 变为 **O(1)**，而 prefill 仍可
> 通过分块并行保持 O(N) 的计算复杂度。

---

## 1. 为什么做这个项目

2025 年以来，线性 / 混合注意力成为大模型推理最前沿的方向之一：

- **Kimi Linear**（2025.10 开源 KDA kernel + vLLM 实现 + 权重，arXiv:2510.26692），
  Kimi K3 全面采用 KDA；
- **Mamba2 / SSD** 把状态空间模型做到"递推与注意力对偶"，NVIDIA 持续发布高性能内核；
- vLLM、SGLang 等已官方支持 hybrid 模型与 O(1) 状态缓存。

本项目的目标是**从数学、内核到系统三层打通**，而不是调用现成 `mamba-ssm` 黑盒。

---

## 2. 支持的层（mixer）

| 层 | 关键机制 | 状态 | 代码 |
| --- | --- | --- | --- |
| **Mamba1** | 选择性 SSM、连续→离散化、并行 scan | `{h, conv}` | `layers/ssm.py` |
| **Mamba2** | 多头 SSD，recurrent ↔ chunkwise 对偶 | `{S, conv}` | `layers/mamba2.py` |
| **DeltaNet** | delta rule（在线误差修正） | `S (H,P,K)` | `layers/deltanet.py` |
| **Gated DeltaNet** | delta rule + 标量遗忘门 + WY/UT 分块 | `S (H,P,K)` | `layers/deltanet.py` |
| **KDA** | Kimi Delta Attention，逐通道门控（DPLR） | `S (H,P,K)` | `layers/kda.py` |
| **SWA** | 滑动窗口因果注意力（GQA），窗口固定 KV | `{k,v,length}` | `layers/attention.py` |

每一层都同时提供：

- **recurrent（循环）**：decode 单步，固定大小状态；
- **chunkwise（分块）**：prefill 并行，块内矩阵运算；
- 二者数学等价，并由测试逐一验证。

---

## 3. 架构总览

```
                         ┌──────────────────────────────┐
 token ids ─▶ embedding  │  × n_layers（pre-norm 残差）  │
                         │   x = x + Mixer(RMSNorm(x))   │
                         └──────────────┬───────────────┘
                                        │ mixer 由 mixer_pattern 决定
              ┌──────────────┬──────────┴─────────┬───────────────┐
        Mamba1/2 SSD    GatedDeltaNet / KDA           SWA
        （状态递推）      （delta rule + 门控）       （窗口注意力）
              │              │                         │
              └──────────────┴──────── 固定大小 state ──┘
                                        │
                          RMSNorm_f ─▶ lm_head ─▶ logits ─▶ 采样/生成
```

生成引擎（`engine.py）：**prefill（整段）→ 采样首 token → decode（逐 token 复用状态）**，
支持贪心 / temperature / top-k / top-p 采样。

---

## 4. 快速开始

### 4.1 安装

```bash
# 仅 PyTorch 路径（无需 nvcc，任何环境可用）
pip install -r requirements.txt
pip install -e .
```

### 4.2 离线 demo（无需下载权重）

```bash
python examples/demo.py
# 或直接用随机小模型生成（输出无意义，仅验证流程）
python scripts/run_generation.py --random --prompt "hello"
```

### 4.3 加载真实 Mamba2 权重

```bash
# 下载（单卡 8GB 建议从 130m / 370m 起步）
python scripts/download_weights.py --model state-spaces/mamba2-130m \
    --out weights/mamba2-130m

# 生成
python scripts/run_generation.py --weights weights/mamba2-130m \
    --prompt "The future of AI is" --max-new 64 --temperature 0.7
```

> 网络受限时可设置镜像：`set HF_ENDPOINT=https://hf-mirror.com`。

### 4.4 Benchmark：线性 scaling 与 O(1) 状态

```bash
python scripts/benchmark.py --lengths 128,256,512,1024
```

典型结果（小尺寸合成模型，CPU）：

| 长度 | prefill (s) | 线性 state | Transformer KV |
| --- | --- | --- | --- |
| 128 | 0.0104 | 32 KB | 512 KB |
| 256 | 0.0211 | 32 KB | 1.0 MB |
| 512 | 0.0391 | 32 KB | 2.0 MB |

prefill 时间随长度线性增长，而线性模型 state 恒定；Transformer KV 随长度线性增长。

### 4.5 运行测试

```bash
python -m pytest tests/ -v
# 18 passed
```

---

## 5. 项目结构

```
linear-hybrid-engine/
├── lhengine/
│   ├── config.py            # 模型/引擎配置、mixer 规格
│   ├── weights.py           # safetensors 真实权重加载 + 随机初始化
│   ├── tokenizer.py         # HF tokenizer / 字节级 fallback
│   ├── engine.py            # prefill+decode 主循环与采样
│   ├── layers/
│   │   ├── common.py        # RMSNorm、因果 depthwise 卷积、门控
│   │   ├── ssm.py           # Mamba1 选择性 SSM
│   │   ├── mamba2.py        # Mamba2 SSD（recurrent/chunkwise）
│   │   ├── deltanet.py      # DeltaNet / Gated DeltaNet（WY/UT）
│   │   ├── kda.py           # KDA（逐通道门控）
│   │   ├── attention.py     # SWA 滑动窗口注意力
│   │   └── block.py         # MixerBlock 统一封装
│   ├── cache/
│   │   └── state_cache.py   # 固定大小状态缓存、fork 分支
│   ├── kernels/
│   │   ├── scan.py          # 纯 PyTorch 并行 scan
│   │   ├── loader.py        # CUDA 扩展加载与回退
│   │   └── csrc/            # 可选 CUDA 内核（scan / SSD）
│   └── models/
│       └── lm.py            # CausalLM 组装（纯线性 / Hybrid）
├── tests/                   # 18 个数值对齐测试
├── scripts/                 # 下载权重 / 生成 / benchmark
├── examples/                # 最小可运行示例
├── docs/                    # 数学笔记 L1–L6
├── requirements.txt
└── setup.py
```

---

## 6. 可选：编译自研 CUDA 内核

需要 NVIDIA GPU 与匹配的 CUDA 工具链（`nvcc`）：

```bash
# Windows (PowerShell)
$env:LHENGINE_CUDA="1"; pip install -e .
# Linux
LHENGINE_CUDA=1 pip install -e .
```

编译后 PyTorch 层会自动优先使用 CUDA 内核；未编译时无缝回退。详见
[`lhengine/kernels/csrc/README.md`](lhengine/kernels/csrc/README.md)。

---

## 7. 数学笔记

完整数学推导见 [`docs/`](docs/README.md)：

1. 连续 SSM → 离散递推（ZOH）
2. 并行扫描（Hillis–Steele / 分块）
3. SSD 对偶：状态递推 ↔ 类注意力
4. Delta rule：在线误差修正
5. WY / UT：delta rule 的分块并行
6. KDA 与 DPLR：逐通道门控

---

## 8. 参考资料

- Kimi Linear 技术报告：<https://arxiv.org/pdf/2510.26692v1>
- Gated DeltaNet（含附录 A WY 推导）：<https://arxiv.org/pdf/2412.06464v3>
- 线性注意力家族统一记号综述：<https://arxiv.org/html/2607.07953>
- NVIDIA cuTile Mamba2 内核：<https://developer.nvidia.com/blog/accelerating-mamba-2-with-high-performance-forward-pass-kernels-in-nvidia-cuda-tile/>
- SGLang hybrid MambaRadixCache：<https://pytorch.ac.cn/blog/hybrid-models-meet-sglang-more-than-full-attention/>
- Compiler-First SSD / O(1) caching：<https://arxiv.org/html/2603.09555>
- mamba-ssm：<https://pypi.org/project/mamba-ssm/>
