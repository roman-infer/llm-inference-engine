// GEMM v0：朴素 kernel，每个线程负责一个输出元素。
//
// C[m,n] = sum_k A[m,k] * B[k,n]
//
// 不使用共享内存，不向量化。这是正确性基准；后续所有版本都必须
// 在 atol/rtol=1e-2 范围内与其（以及 C++ 参考实现）一致。

#include "cuda_llm/gemm.h"

namespace cudallm {

namespace {

constexpr int kTileX = 16;
constexpr int kTileY = 16;

__global__ void gemm_v0_kernel(const __half *__restrict__ A,
                               const __half *__restrict__ B,
                               __half *__restrict__ C, int M, int N, int K) {
  const int n = blockIdx.x * kTileX + threadIdx.x;
  const int m = blockIdx.y * kTileY + threadIdx.y;

  if (m < M && n < N) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
      acc += __half2float(A[m * K + k]) * __half2float(B[k * N + n]);
    }
    C[m * N + n] = __float2half(acc);
  }
}

}  // namespace

void gemm_launch_v0(const __half *A, const __half *B, __half *C, int M, int N,
                    int K, cudaStream_t stream) {
  dim3 block(kTileX, kTileY);
  dim3 grid(ceil_div(N, kTileX), ceil_div(M, kTileY));
  CUDA_LAUNCH(gemm_v0_kernel, grid, block, 0, stream, A, B, C, M, N, K);
}

}  // namespace cudallm
