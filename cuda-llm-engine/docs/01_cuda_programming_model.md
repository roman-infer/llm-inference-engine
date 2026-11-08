# 01 · CUDA 编程模型

> 目标：理解一段 CUDA 代码是如何被映射到 GPU 硬件上并行执行的。本引擎所有
> kernel 都只用到这套最基础的概念，不依赖任何封装库。

## 1. 线程层次：Thread / Warp / Block / Grid

```
Grid
└── Block (可三维索引 blockIdx)
    └── Warp = 32 个 Thread（硬件调度单位，不显式声明）
        └── Thread (threadIdx)
```

- **Thread**：最小执行单元，拥有自己的寄存器与 PC。
- **Warp**：32 个线程为一组，**锁步（lock-step）执行同一条指令**（SIMT）。
  分支 `if` 会让走不同路径的 lane 串行执行，称为 warp divergence。
- **Block**：一组线程（本引擎常用 256/1024），共享同一块 Shared Memory，可
  以用 `__syncthreads()` 做块内同步。Block 之间**不能**直接同步。
- **Grid**：一次 kernel launch 的所有 block。Block 之间相互独立，由 GPU 调
  度到不同 SM，可以任意顺序执行——这是 GPU 可扩展到任意规模的关键。

索引示例（二维 GEMM）：

```cpp
int n = blockIdx.x * blockDim.x + threadIdx.x;
int m = blockIdx.y * blockDim.y + threadIdx.y;
```

## 2. 硬件映射：SM

- GPU 由若干 **SM（Streaming Multiprocessor）** 组成（如 RTX 4060 Laptop
  sm_89，可在 `nvidia-smi`/设备属性查看 SM 数量）。
- Block 被整个分配到一个 SM；一个 SM 上可同时驻留多个 block/warp，由硬件
  在 warp 停顿（等内存/依赖）时快速切换到就绪 warp，用并行掩盖延迟。
- **Occupancy** = SM 上活跃 warp 数 / 最大可驻留 warp 数。它受寄存器与
  shared memory 用量限制。Occupancy 不是越高越好，但过低通常意味着延迟
  无法被掩盖。

## 3. Host 与 Device

```cpp
// __global__：host 调用、device 执行，返回 void，异步
__global__ void kernel(...) { ... }

kernel<<<grid, block, shm_bytes, stream>>>(args);   // launch
cudaDeviceSynchronize();                            // 等待全部完成
```

- launch 是**异步**的：控制权立刻回到 CPU。需要结果前必须同步（event /
  stream synchronize）。
- `__device__` 函数在设备上编译、由 kernel 调用；`__host__` 是普通函数。
- nvcc 把 kernel 编译成 SASS（实际机器码），同一份代码可用
  `-gencode arch=compute_89` 等指定目标架构（见 CMakeLists）。

## 4. 同步与流（Stream）

- **Stream** 是一条异步任务队列；同一 stream 内 kernel 按入队顺序执行，不
  同 stream 可以重叠。本引擎 `Context` 持有一个非阻塞 stream，所有算子都
  显式接收它，保证调用顺序确定。
- 同步粒度由粗到细：`cudaDeviceSynchronize` > `cudaStreamSynchronize` >
  `cudaEventSynchronize`。性能计时用最轻的 Event（见 `EventTimer`）。

## 5. 本引擎中的对应关系

| 概念 | 代码位置 |
| --- | --- |
| launch / 错误检查 | `include/cuda_llm/cuda_utils.h` 的 `CUDA_LAUNCH` |
| 运行时 stream | `src/core/context.cu` |
| 一线程一元素（入门） | `src/kernels/gemm_v0.cu` |
| warp 协作归约 | `attention.cu`、`sampling.cu` |
| block 内共享与同步 | `gemm_v2.cu` |

## 6. 动手实验

1. 把 `gemm_v0` 的 block 从 16×16 改成 32×32、8×8，观察 grid 数量与耗时。
2. 在 kernel 里打印 `threadIdx/warpId`，确认一个 warp 内 lane 的编号连续。
3. 故意写一个发散分支（按 `threadIdx.x % 2`），用 Nsight Compute 看
   divergence 指标变化。
