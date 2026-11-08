# 06 · 学习路径

> 一条"能跑通 → 算得对 → 读得懂 → 跑得快"的 CUDA 推理学习路线，按依赖
> 顺序排列，每一步都有可运行代码与可验证目标。

## 阶段 0：环境与心智模型（1 周）

- 安装 CUDA Toolkit、CMake、编译器（Windows：VS + CUDA；Linux：gcc + CUDA）。
- 阅读文档 01、02；跑通空 kernel、线程索引打印、vector add。
- 验收：能解释 grid/block/warp、合并访存、bank conflict。

## 阶段 1：GEMM 主线（核心，2–4 周）

按 v0→v4 逐级实现，**每级先写 CPU 参考与数值比对，再谈性能**：

1. v0 naive：一线程一元素。
2. v1 half2/向量化：理解对齐与合并。
3. v2 shared tiling：padding、寄存器累加、unroll。
4. v3 epilogue 融合：理解归约与"少一次访存"。
5. v4 GEMV：理解带宽受限形状与权重布局。

验收：
- `test_gemm` 全部通过（atol=rtol=1e-2）；
- `bench_gemm` 能解释每一级为什么变快/变慢；
- 会用 CUDA Event 计时并计算 TFLOPS。

## 阶段 2：算子与模型串联（2–3 周）

- 阅读文档 04、05；实现 norm/rope/activation/embedding/sampling。
- 串 GPT-2：先用 `make_tiny.py` 随机权重跑通形状，再用
  `convert_gpt2.py` 导出真实 GPT-2 124M 完成文本生成。
- 串 LLaMA 风格（GQA + SwiGLU + RoPE）。

验收：不看教程能默画两种 block 结构；能定位一个数值偏差出在哪个算子。

## 阶段 3：性能分析方法（1 周）

- 学习 Nsight Systems（时间线/launch 开销）与 Nsight Compute
  （occupancy、stall reason、带宽、bank conflict、warp 利用率）。
- 建立 roofline：查目标架构各精度峰值算力/带宽，计算每个 kernel 的
  峰值达成率，判断它是 compute-bound 还是 memory-bound。
- 把指标采集与阈值接入 CI（性能回归）。

## 阶段 4：进阶专题（按兴趣，长期）

| 方向 | 关键技术 |
| --- | --- |
| Tensor Core | WMMA、`mma.sync`、FP32 累加、权重 interleave |
| 异步流水线 | `cp.async`、multi-stage buffer、TMA、cluster、warp specialization |
| Attention | 在线 softmax（FlashAttention 风格）、FlashDecoding |
| 量化 | W8A8（INT8 Tensor Core）、W4A16、FP8、KV cache 量化 |
| 调度 | CUDA Graph、PagedAttention、continuous batching、chunked prefill |
| 扩展 | 多 GPU 张量并行、MoE 分组 GEMM、prefix cache、投机解码 |
| 前沿硬件 | Hopper FP8/TMA、Blackwell FP4/MX |

## 推荐学习资料类型（不依赖具体链接）

- 官方 CUDA C++ Programming Guide / Best Practices Guide（权威手册）。
- 各代架构 Whitepaper（理解 Tensor Core、内存单元演进）。
- 经典开源 GEMM/Attention 实现的**设计文档**（先读算法，再对照自己写，
  不要直接抄库——与本引擎"不依赖官方计算库"的定位一致）。

## 常见误区

1. 只比"相对 naive 加速比" → 应看峰值达成率。
2. 一上来追新硬件特性 → 先把合并访存/tiling 做扎实。
3. 忽略数值正确性 → 每个 kernel 都要有 CPU 参考与容差。
4. 在热循环里 malloc → 预分配 + 池化。
5. 只看平均时间 → 用中位数/多次采样，关注抖动。

## 一句话路线

**先用最简单的实现保证算得对，再按"访存效率 → 数据复用 → 减少往返 →
形状特化 → 专用硬件 → 系统调度"的顺序逐级逼近峰值，每一步都可测量、
可验证、可回退。**
