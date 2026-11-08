# cuda-llm-engine

从零实现的 C++/CUDA 原生 Transformer 推理引擎。**不依赖 cuBLAS / cuDNN / FlashAttention**，
所有 GPU 计算代码手写，配套 C++/CPU 参考实现逐元素比对数值（atol=rtol=1e-2）。

目标：**可运行、数值正确、性能优化、文档清晰**——可单步复现矩阵乘法在 GPU 上的完整执行过程，
且每一步优化都有实测数字，不是只有理论推导。

---

## 实测数据

环境：RTX 4060 Laptop（sm_89, 24 SM, 8GB）/ CUDA 13.4 / `-O3` Release / CUDA Event 计时，取中位数。
复现：`./build/bench_gemm --json results/bench_gemm.json`

### GEMM 五级优化（FP16 输入，FP32 累加）

| 形状 (M×N×K) | v0 naive | v1 向量化 | v2 分块 + 寄存器分块 |
| --- | --- | --- | --- |
| 1024×4096×4096 | 36.85 ms / 0.93 TFLOPS | 17.98 ms / 1.91（**2.05×**） | 6.60 ms / 5.21（**5.59×**） |
| 128×4096×4096 | 4.78 ms / 0.90 | 2.20 ms / 1.95（2.17×） | 0.96 ms / 4.49（**5.00×**） |
| 2048×2048×2048 | 17.96 ms / 0.96 | 8.87 ms / 1.94（2.02×） | 3.37 ms / 5.09（**5.33×**） |
| 128×8×8192（退化） | 0.294 ms / 0.06 | 0.207 ms / 0.08（1.42×） | 0.537 ms / 0.03（**0.55×**） |

两点值得单独说明：

* **v2 在真实形状上快 5×，在 N=8 这种极窄形状上反而慢 45%。** 分块的价值来自复用，
  BN=64 的块在 N=8 时浪费 87.5% 的算力——所以 decode 阶段必须换成 v4 GEMV，
  而不是硬用 GEMM。**“按形状特化”是这个项目最重要的一条工程结论。**
* v2 的加速来自**寄存器分块**：每个线程算 4×4 个输出，每读一对共享内存分量能摊到 4 次
  FMA 上。只做 shared tiling、不给每线程多个输出，是拿不到加速的（旧版就是这么写的，
  实测比 naive 还慢 2.5×，见下方“已修复的问题”）。

### decode（M=1，GEMV）

| 形状 (N×K) | 权重体积 | 延迟 | 等效带宽 |
| --- | --- | --- | --- |
| 4096×4096 | 33.5 MB | 0.030 ms | 1130 GB/s（反复读同一份权重，基本命中 L2） |
| 16384×4096 | 134 MB | 0.546 ms | **246 GB/s ≈ HBM 峰值** |

同一份 33.5MB 权重在重复计时下会命中 L2，带宽数字会“虚高”4 倍；换成 134MB 才落回 HBM 峰值。
**报带宽时不交代工作集大小，数字没有意义**——这是 benchmark 最容易产生误导的地方。

### Attention：三趟参考实现 vs 在线 softmax

配置 H=12, KVH=12, d=64；`./build/bench_attention --json results/bench_attention.json`

| seq | kv | 三趟 | 在线 softmax | 三趟需物化的分数矩阵 |
| --- | --- | --- | --- | --- |
| 1 | 4096 | **0.805 ms** | 1.119 ms | 0.10 MB |
| 128 | 128 | 0.145 ms | **0.060 ms** | 0.39 MB |
| 128 | 1024 | 1.006 ms | **0.773 ms** | 3.15 MB |
| 128 | 4096 | 6.170 ms | **3.164 ms** | 12.58 MB |

结论：**在线 softmax 在 prefill 快 2–2.4× 且省掉 O(S·KV) 的中间矩阵；但在 decode（seq=1）
反而慢 39%**——每行只有一个 warp，并行度铺不开。这正是生产系统里 prefill 用 FlashAttention、
decode 要单独用 FlashDecoding / split-K 的原因，而不是依赖单一 kernel 覆盖全部场景。

---

## 已修复的三个真实问题

这三个 bug 都表现为“程序可以运行但结果错误”，而且互相掩盖，只有对着参考实现逐层比对才会暴露。
记录在这里，因为它们比功能清单更能体现工程过程。

1. **HuggingFace GPT-2 的 Conv1D 权重是 `[in, out]`，不是 `[out, in]`。**
   加载器按 PyTorch `nn.Linear` 的约定转置，导致 `qkv` 只算出 768 个输出（应为 2304 个），
   其余通道是未初始化内存；`fc` 同理。修复：新增 `make_linear_conv1d`，直接把存储布局
   当作 GEMM 需要的 `wt[K,N]`，再转置出 GEMV 用的 `w[N,K]`。

2. **`Tensor` 不带 stride，`slice(1, ...)` 之后再 `view()` 会按连续布局误读。**
   `qkv[n, 3E]` 切出 `q/k/v` 后，第 2 行开始读到的其实是上一行的 k、v。
   修复：新增 `op::split_qkv`，显式拷贝成三个连续缓冲（多一次拷贝，换正确性）。

3. **注意力分数漏了 `1/sqrt(d)` 缩放。** 未缩放时 softmax 接近 one-hot，生成会退化成
   “同一 token 无限重复”。修复：缩放放进 kernel，调用方不用预处理 q。

修复前后对比（同一个 prompt、同一份 124M 权重、greedy）：

```
修复前：The meaning of life is impersonsoDeliveryDatesoDeliveryDatesoDeliveryDate...
修复后：The meaning of life is not the same as the meaning of death.
```

另外修掉一个隐蔽的全局状态泄漏：`logger.cu` 里的 `std::setfill('0')` 是粘性状态，
会污染之后所有 `std::setw`，让 benchmark 表格被补成 `000000v0_naive...`。

---

## 特性

### GEMM 五级优化

| 级别 | 内容 | 关键点 |
| --- | --- | --- |
| v0 | naive，一线程一元素 | 正确性基线 |
| v1 | half2 向量化 | 合并访存、每线程 2 输出、FP32 累加 |
| v2 | Shared Tiling + 寄存器分块 | 64×64×16 块、每线程 4×4 输出、`Asub[][BM+8]` 缓解 bank conflict |
| v3 | Epilogue Fusion | GEMM + Bias + RMSNorm 单 kernel，省 2MN half 的 HBM 往返 |
| v4 | GEMV + KV Cache | decode 专用，warp per output，M=1 有效占用 0.78%→100% |

### Transformer 全链路算子

LayerNorm / RMSNorm、MHA/MQA/GQA + RoPE + KV Cache、SiLU / GELU、
Top-K / Top-P / Temperature 采样、Embedding、F32↔F16 转换、2D transpose、逐元素 add/mul；
权重以 **safetensors** 加载；串成 **GPT-2 124M**（12 层 / 768 hidden / FP16 / max_seq=1024）
与 **LLaMA 风格**模型，完成真实文本生成。

### C++ 工程体系

Tensor（多维 + 内存管理）、Allocator（池化缓存）、Checkpoint（手写 JSON 解析）、
Context（CUDA Stream + workspace）、BPETokenizer、Logger；CMake 构建；单元测试覆盖
GEMM v0–v4、7 类算子，以及 attention 两条路径与 CPU 参考的三方对拍。

---

## 快速开始

需要 CUDA Toolkit（11.x+，推荐 12.x/13.x）、CMake ≥ 3.18、C++17 编译器。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j
ctest --test-dir build --output-on-failure     # test_gemm + test_ops
```

### 准备权重

```bash
# A. 离线随机 tiny 模型（无需网络，验证全链路，输出为乱码）
python tools/make_tiny.py --arch gpt2  --out models/tiny-gpt2
python tools/make_tiny.py --arch llama --out models/tiny-llama

# B. 真实 GPT-2 124M
pip install torch transformers
python tools/convert_gpt2.py --model gpt2 --out models/gpt2
```

### 运行

```bash
./build/generate --arch gpt2 --ckpt models/gpt2/model.safetensors \
                 --tok models/gpt2 --prompt "The meaning of life is" \
                 --max_new 32 --temp 0 --top_k 50 --top_p 0.95

./build/bench_gemm      --json results/bench_gemm.json
./build/bench_attention --json results/bench_attention.json
```

---

## 目录结构

```
cuda-llm-engine/
├── include/cuda_llm/      # tensor / allocator / context / checkpoint / tokenizer /
│                          # ops / gemm / model / json_min ...
├── src/
│   ├── core/              # 基础设施实现
│   ├── kernels/           # GEMM v0–v4 + 全部算子 + attention（三趟 / 在线 softmax）
│   ├── models/            # gpt2 / llama
│   └── apps/              # generate / bench_gemm / bench_attention
├── tests/                 # CPU 参考 + 单元测试（含 attention 两条路径对拍）
├── tools/                 # 权重转换 / 离线 tiny 模型
├── results/               # 基准输出 JSON（可复现的数字来源）
└── docs/                  # 6 篇原理笔记
```

---

## 边界（没做什么，以及为什么）

* **没有 Tensor Core。** 全部 kernel 走 CUDA Core，5 TFLOPS 是这套实现的量级上限；
  再往上必须用 WMMA / `mma.sync`，这是明确的下一步而不是遗漏。
* **attention 的 decode 路径还慢。** seq=1 时在线版本比三趟慢 39%，需要 FlashDecoding /
  split-K 把 kv 维切开并行，当前版本没做。
* **没有 batching / PagedAttention。** 单序列、连续 KV cache，`generate` 一次只跑一条；
  多请求调度与分页 KV 是另一个项目（deepseek-serving）覆盖的内容。
* **没有量化。** 全部 FP16，无 INT8/FP8 路径。
* **v3 的 epilogue 融合只在部分形状赢**：一个 block 必须覆盖整行才能做 RMSNorm 统计，
  BM=4 限制了 M 维并行——它是“省 HBM 往返”的收益，不是算力收益。

## 后续优化路线

* v5 Tensor Core GEMM（WMMA / `mma.sync` m16n8k16）+ cp.async 多级流水线
* FlashDecoding / split-K 解决 decode 侧并行度
* tile autotune、persistent kernel
* 量化（W8A8 / W4A16 / FP8，KV cache 量化）
* CUDA Graph、PagedAttention、continuous batching
* 多 GPU 张量并行、MoE、prefix cache

## 文档

1. [CUDA 编程模型](docs/01_cuda_programming_model.md)
2. [GPU 内存层次](docs/02_memory_hierarchy.md)
3. [GEMM 优化路线](docs/03_gemm_optimization.md)
4. [Transformer 推理流程](docs/04_transformer_inference.md)
5. [KV Cache 机制](docs/05_kv_cache.md)
6. [学习路径](docs/06_learning_path.md)

## 许可

MIT，见 [LICENSE](LICENSE)。
