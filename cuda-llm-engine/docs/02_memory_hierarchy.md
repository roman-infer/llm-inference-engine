# 02 · GPU 内存层次

> 目标：建立"算得慢 vs 搬得慢"的直觉。GPU 优化绝大多数情况下是内存优化。

## 1. 层次总览

```
寄存器 Registers      ~ 每线程私有，最快，数量有限（每 SM 64K 个 32-bit）
Shared Memory / L1   ~ 每 SM，低延迟，~ 可配置（数十~上百 KB）
L2 Cache             ~ 全芯片共享，数 MB
HBM（显存）          ~ 全局，容量大，带宽高但延迟高（数百周期）
```

经验数量级（不同架构有差异，重点是相对关系）：

| 介质 | 相对延迟 | 带宽 | 作用域 |
| --- | --- | --- | --- |
| 寄存器 | ~1 | — | 线程 |
| Shared | ~几十周期 | 很高 | Block/SM |
| L2 | ~上百周期 | 高 | 全 GPU |
| HBM | ~数百周期 | 数 TB/s | 全 GPU |

## 2. 合并访存（Coalescing）

一个 warp 的 32 个 lane 如果访问**连续且对齐**的地址，硬件会把 32 次访问
合并成少数几个内存事务；反之（跨步、随机）会产生大量浪费事务。

```cpp
// 好：lane i 读 base[i]（连续）
float v = arr[blockIdx.x * blockDim.x + threadIdx.x];

// 坏：lane i 读 base[i * stride]（大跨步）
float v = arr[threadIdx.x * 1024];
```

本引擎中：
- `gemm_v2` 加载 A tile 时让线程沿 K 连续读，写回 C 时让 warp 沿 N 连续写。
- `attention` scores kernel 中 lane 沿 head_dim 连续读 K，保证合并。

## 3. Shared Memory 与 Bank Conflict

Shared memory 被切成 32 个 bank，warp 中每个 lane 各访问一个 bank 时一个
周期可全部完成。

- 多个 lane 访问**同一 bank 不同地址** → bank conflict，请求被串行化。
- 多个 lane 访问**完全相同地址** → 广播，不冲突。

经典解法是 **padding**：

```cpp
// 行宽为 BK 时，warp 沿行读同一列：bank = (m*BK+k)%32 周期性重复
__shared__ half Asub[BM][BK + 1];   // 加一列，gcd(BK+1,32)=1 → 零冲突
```

这正是 `gemm_v2.cu` 中 `Asub[BK+1]` 的由来。

## 4. 数据复用与 Tiling

矩阵乘中每个 A/B 元素被重复使用多次。直接从 HBM 读会让程序完全受带宽
限制。Tiling 的思想：

1. 把当前要算的输出子块所需的 A/B 小块搬到 shared memory；
2. block 内所有线程复用这块数据完成子块计算；
3. 沿 K 迭代，直到覆盖整个矩阵。

数据复用次数随 tile 增大而提高，但 tile 受 shared memory / 寄存器容量
约束——这是 tile 形状（BM×BN×BK）需要权衡与 autotune 的原因。

## 5. 异步拷贝（进阶，了解）

- sm_80+ 提供 `cp.async`：global→shared 的拷贝**不占用寄存器、不阻塞
  计算**，配合 double/multi-stage buffer 让"搬下一块"和"算当前块"重叠。
- sm_90（Hopper）提供 TMA（Tensor Memory Accelerator）硬件单元与线程块
  cluster，可由单个线程发起大块张量搬运。

本引擎当前版本用同步加载（`__syncthreads`），异步流水线是规划中的
下一步（见 03 文档末尾与 README 路线图）。

## 6. 显存分配

- `cudaMalloc/cudaFree` 带同步开销，不应出现在逐 token 的热循环里。
- 本引擎 `CachingAllocator` 把释放的块保留在按大小排序的 free-list 中
  复用；模型构建时一次性分配权重/KV cache/scratch，forward 中不再向
  驱动申请内存。

## 7. 动手实验

1. 用 Nsight Compute 查看 `gemm_v0` 与 `gemm_v2` 的 Global Memory
   Throughput、Shared Memory Bank Conflicts。
2. 去掉 `gemm_v2` 的 `+1` padding，对比 bank conflict 指标。
3. 计算一次 GEMM 的最小字节搬运量（`A+B`），与实测带宽相乘估算理论
   最短时间，和实测对比（roofline 的雏形）。
