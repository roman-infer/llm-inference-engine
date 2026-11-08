# 03 · GEMM 优化路线

> 主线：v0 naive → v1 向量化 → v2 Shared Tiling → v3 Epilogue Fusion →
> v4 GEMV + KV Cache。每一级都保持相同数值语义，可用
> `bench_gemm` 量化收益、用 `test_gemm` 验证正确性。

> **2026-10 修订**：v2 已从「BM=128 / BN=8 / BK=8、每线程 1 个输出」改成
> 「64×64×16、每线程 4×4 个输出」的寄存器分块版本。旧版只做了 shared tiling、
> 没有给每线程多个输出，实测在 M=1024/K=4096 上比 naive 慢 2.5×；新版同一形状
> 快 5.59×（6.60 ms / 5.21 TFLOPS）。下面 v2 一节保留旧设计的分析，作为反例，
> 新实现见 `src/kernels/gemm_v2.cu` 顶部注释。

约定：`C[M,N] = A[M,K] * B[K,N]`，row-major，FP16；计算统一在 FP32 累加。

## v0 · Naive（一线程一元素）

```cpp
for (k ...) acc += A[m*K+k] * B[k*N+n];
```

- 无 shared、无向量化；访存不合并、数据零复用。
- 定位：**正确性基线**。后续所有版本都与它及 CPU 参考比对
  （atol=rtol=1e-2）。

## v1 · float4 / half2 向量化

- 每线程持有 1×2 输出（列 n、n+1），K 每次推进 2：
  - half2 载入 A 的两个元素，分别广播；
  - half2 载入 B 的两行两列，`__hfma2` 累加。
- 相邻线程读相邻 half2，全局访存合并成 float4 级事务。
- 要点：地址对齐（K/N 为偶，tensor 基址 256 对齐），奇数维走标量 tail。

## v2 · Shared Memory Tiling

- block tile `BM×BN = 128×8`，K tile `BK=8`。
- 协作把 A/B tile 搬入 shared；每个线程在**寄存器**中持有输出元素，沿 K
  tile 累加（warp 级寄存器累加），内层 BK 循环 `#pragma unroll`。
- `Asub[BM][BK+1]` padding 规避 bank conflict（见 02 文档）。
- 线程映射 x=n、y=m：warp 沿 N 连续 → C 写回合并，Asub 读取为广播。

> 说明：128×8×8 是便于讲解的配置（每个结构都很小、容易单步跟踪）。生产
> 级实现通常用 128×128×32 量级并做 autotune；BN/BK 偏小意味着数据复用
> 与并行度都有提升空间。

## v3 · Epilogue Fusion（GEMM + Bias + RMSNorm）

- 一个 block 覆盖 BM 行 × **全部 N 列**：GEMM 结果完整驻留 shared memory。
- 加 bias → 行内归约平方和 → RMSNorm（乘 gamma）→ **只写一次** HBM。
- 关键约束：RMSNorm 需要整行统计量，因此 block 必须覆盖整行；B tile 的
  协作加载必须所有线程参与（否则 `__syncthreads` 死锁）。
- 收益：消除 GEMM 结果"写出再读回"的一次完整往返。

## v4 · GEMV + KV Cache（decode 专用）

- decode 阶段输入只有一个 token（M=1），GEMM 退化为 GEMV，**完全受带宽
  限制**。
- 权重用 `w[N,K]`（out×in）布局，每个输出行在 HBM 中连续；**一个 warp
  负责一个输出**，lane 沿 K 步长 64（half2），warp shuffle 归约。
- 产出的 K/V 随即通过 `kv_append` 写入 KV cache（见 05 文档）。

## 基准测试方法

- 统一用 CUDA Event：warmup + N 次计时，取中位数（见
  `gemm_dispatch.cu`、`bench_gemm.cu`）。
- 报三类数字：**ms**、**TFLOPS = 2MNK/(ms·1e9)**、相对 v0 的加速比。
- 更严谨的口径是**峰值达成率**：TFLOPS / 该 GPU 对应精度的理论峰值
  （roofline）。"相对 naive 加速比"会随基线变弱而虚高，峰值达成率才是
  跨硬件可比的指标。

## 演进的内在逻辑

```
v0  正确但什么都不优化
 v1 解决"单次访存效率"（向量化/合并）
 v2 解决"数据复用"（shared tiling + 寄存器累加）
 v3 解决"结果的重复读写"（epilogue 融合）
 v4 解决"decode 形状退化"（GEMV + 权重布局）
```

## 下一步（规划中，尚未实现）

1. **v5 Tensor Core GEMM**：WMMA / `mma.sync m16n8k16`，FP16 输入 FP32
   累加——Tensor Core 峰值数倍于 CUDA Core。
2. **cp.async 多级流水线**（sm_80+），TMA/cluster（sm_90）。
3. tile 形状 autotune、persistent kernel、split-K。
4. 在线 softmax 的 attention（见 04/05 文档）。
