// GEMM v3：Epilogue 融合 —— 在单个 kernel 中完成 GEMM + Bias + RMSNorm。
//
// 一个 block 处理 BM 行跨全部 N 列，因此矩阵乘完成后整行数据都在
// 共享内存中，可以直接归一化而无需将原始 GEMM 结果写入全局内存：
//
//   C[m,:] = RMSNorm(A[m,:] * B + bias) * gamma
//
// 每个 K 分块（BK=2）的流水线：
//   1. 协作加载 B 分块 [BK,N]（所有线程参与）和 A 标量 [BM,BK]
//   2. 累加到共享内存中的 Cacc[BM,N]
// 所有 K 分块完成后：加 bias -> 逐行平方和归约 -> RMSNorm ->
// 每行一次合并的全局写回。
//
// 共享内存（N=3072）：约 36 KiB；更大 N 时可申请至多 96 KiB。

#include "cuda_llm/gemm.h"

namespace cudallm {

namespace {

constexpr int BM = 4;
constexpr int BK = 2;
constexpr int BLOCK = 256;
constexpr int ROW_THREADS = BLOCK / BM;  // 每行 64 个线程（2 个 warp）

__global__ void gemm_v3_kernel(const __half *__restrict__ A,
                               const __half *__restrict__ B,
                               const __half *__restrict__ bias,
                               const __half *__restrict__ gamma,
                               __half *__restrict__ C, int M, int N, int K,
                               float eps) {
  extern __shared__ __half smem[];
  __half *Bt = smem;                  // [BK][N]
  __half *At = Bt + BK * N;           // [BM][BK]
  __half *Cacc = At + BM * BK;        // [BM][N]

  const int row = threadIdx.x / ROW_THREADS;  // 0..BM-1
  const int rt = threadIdx.x % ROW_THREADS;   // 0..63
  const int m = blockIdx.x * BM + row;

  for (int idx = threadIdx.x; idx < BM * N; idx += BLOCK) {
    Cacc[idx] = __float2half(0.0f);
  }
  __syncthreads();

  for (int bk = 0; bk < K; bk += BK) {
    // B 分块：所有线程都必须参与（沿 n 合并访存）。
    for (int idx = threadIdx.x; idx < BK * N; idx += BLOCK) {
      const int kk = idx / N;
      const int n = idx % N;
      const int gk = bk + kk;
      Bt[idx] = (gk < K) ? B[gk * N + n] : __float2half(0.0f);
    }
#pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
      const int gk = bk + kk;
      At[row * BK + kk] =
          (m < M && gk < K) ? A[m * K + gk] : __float2half(0.0f);
    }
    __syncthreads();

    if (m < M) {
      for (int n = rt; n < N; n += ROW_THREADS) {
        float acc = __half2float(Cacc[row * N + n]);
#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
          acc += __half2float(At[row * BK + kk]) *
                 __half2float(Bt[kk * N + n]);
        }
        Cacc[row * N + n] = __float2half(acc);
      }
    }
    __syncthreads();
  }

  if (m < M) {
    // 加 bias。
    for (int n = rt; n < N; n += ROW_THREADS) {
      float v = __half2float(Cacc[row * N + n]) +
                __half2float(bias[n]);
      Cacc[row * N + n] = __float2half(v);
    }
    __syncthreads();

    // 逐行平方和。
    float local = 0.0f;
    for (int n = rt; n < N; n += ROW_THREADS) {
      float v = __half2float(Cacc[row * N + n]);
      local += v * v;
    }
    for (int off = 16; off > 0; off >>= 1) {
      local += __shfl_down_sync(0xffffffffu, local, off);
    }

    __shared__ float warp_sum[BM * 2];
    __shared__ float row_ss[BM];
    const int warp_global = threadIdx.x / 32;
    const int warp_in_row = (threadIdx.x % ROW_THREADS) / 32;
    if ((threadIdx.x % 32) == 0) warp_sum[warp_global] = local;
    __syncthreads();
    if (warp_in_row == 0 && (threadIdx.x % 32) == 0) {
      row_ss[row] = warp_sum[row * 2] + warp_sum[row * 2 + 1];
    }
    __syncthreads();

    const float mean_sq = row_ss[row] / static_cast<float>(N);
    const float scale = rsqrtf(mean_sq + eps);

    for (int n = rt; n < N; n += ROW_THREADS) {
      float v = __half2float(Cacc[row * N + n]) * scale *
                __half2float(gamma[n]);
      C[m * N + n] = __float2half(v);
    }
  }
}

}  // namespace

void gemm_bias_rmsnorm(const __half *A, const __half *B, const __half *bias,
                       const __half *gamma, __half *C, int M, int N, int K,
                       float eps, cudaStream_t stream) {
  const dim3 grid(ceil_div(M, BM));
  const dim3 block(BLOCK);
  const size_t shm =
      (BK * N + BM * BK + BM * N) * sizeof(__half);

  CUDA_CHECK(cudaFuncSetAttribute(
      gemm_v3_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      96 * 1024));
  CUDA_LAUNCH(gemm_v3_kernel, grid, block, shm, stream, A, B, bias, gamma, C,
              M, N, K, eps);
}

}  // namespace cudallm
