#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "cuda_llm/cuda_utils.h"  // CUDA_CHECK / CUDA_LAUNCH

namespace cudallm {

enum class GemmVersion {
  kV0Naive = 0,
  kV1Vectorized = 1,
  kV2Tiled = 2,
  kV3EpilogueFusion = 3,
  kV4GemvKvCache = 4,
};

const char *gemm_version_name(GemmVersion version);

// ---------------------------------------------------------------------------
// 标准 GEMM：C[M,N] = A[M,K] * B[K,N]，行主序，FP16。
// v0..v2 使用相同的函数签名，以便互换对比和 benchmark。
// ---------------------------------------------------------------------------
void gemm_launch_v0(const __half *A, const __half *B, __half *C, int M, int N,
                    int K, cudaStream_t stream);
void gemm_launch_v1(const __half *A, const __half *B, __half *C, int M, int N,
                    int K, cudaStream_t stream);
void gemm_launch_v2(const __half *A, const __half *B, __half *C, int M, int N,
                    int K, cudaStream_t stream);

// 供模型 / benchmark 使用的分发辅助函数。
void gemm_launch(GemmVersion version, const __half *A, const __half *B,
                 __half *C, int M, int N, int K, cudaStream_t stream);

// ---------------------------------------------------------------------------
// v3：融合 GEMM + Bias + RMSNorm epilogue。
//   C[m,:] = RMSNorm(A[m,:] * B + bias) * gamma
// GEMM 结果在共享内存中累加并归一化后，仅写入一次全局内存
// （原始矩阵乘结果从不触碰全局内存）。
// ---------------------------------------------------------------------------
void gemm_bias_rmsnorm(const __half *A, const __half *B, const __half *bias,
                       const __half *gamma, __half *C, int M, int N, int K,
                       float eps, cudaStream_t stream);

// ---------------------------------------------------------------------------
// v4：decode GEMV。权重布局为 w[N,K]（输出 x 输入）：
//   y[n] = sum_k x[k] * w[n][k]
// 一个 warp 计算一个输出 n，从 HBM 流式读取权重行。
// ---------------------------------------------------------------------------
void gemv_launch(const __half *x, const __half *w, __half *y, int K, int N,
                 cudaStream_t stream);

// ---------------------------------------------------------------------------
// 性能测试（CUDA events，预热 + 多次迭代；返回中位数毫秒数）。
// ---------------------------------------------------------------------------
float gemm_benchmark_ms(GemmVersion version, const __half *A, const __half *B,
                        __half *C, int M, int N, int K, cudaStream_t stream,
                        int iters = 20);
float gemv_benchmark_ms(const __half *x, const __half *w, __half *y, int K,
                        int N, cudaStream_t stream, int iters = 50);

}  // namespace cudallm
