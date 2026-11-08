// GEMM v1：使用 half2（以及 float4 宽度加载）的向量化内存访问。
//
// 每个线程负责 C 的一个 1x2 分块（列 n, n+1），每次沿 K 走两个元素：
//
//   a2 = (A[m,k], A[m,k+1])
//   b0 = (B[k,n], B[k,n+1])      b1 = (B[k+1,n], B[k+1,n+1])
//   acc += broadcast(a0) * b0 + broadcast(a1) * b1
//
// 所有加载都是 half2（4 字节）；相邻线程读取相邻的 half2，
// 因此全局加载合并为 float4 大小的访存事务。

#include "cuda_llm/gemm.h"

namespace cudallm {

namespace {

constexpr int kThreadsX = 32;  // 32 个线程 -> 每个 block 64 个输出列
constexpr int kThreadsY = 8;

__global__ void gemm_v1_kernel(const __half *__restrict__ A,
                               const __half *__restrict__ B,
                               __half *__restrict__ C, int M, int N, int K) {
  const int m = blockIdx.y * kThreadsY + threadIdx.y;
  const int n2 = (blockIdx.x * kThreadsX + threadIdx.x) * 2;

  if (m >= M) return;

  if (n2 + 1 < N) {
    // FP32 累加器：half2 存储，float 累加（与其他所有 GEMM 版本
    // 遵循的"FP16 存储、FP32 累加"规则一致；__hfma2 会在 FP16 中
    // 累加，误差会超出容差范围）。
    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (int k = 0; k < K; k += 2) {
      const __half2 a2 =
          *reinterpret_cast<const __half2 *>(&A[m * K + k]);
      const __half2 b0 =
          *reinterpret_cast<const __half2 *>(&B[k * N + n2]);
      const __half2 b1 = __halves2half2(
          B[(k + 1) * N + n2],
          B[(k + 1) * N + n2 + 1]);

      const float a0 = __half2float(__low2half(a2));
      const float a1 = __half2float(__high2half(a2));
      acc0 += a0 * __half2float(__low2half(b0));
      acc0 += a1 * __half2float(__low2half(b1));
      acc1 += a0 * __half2float(__high2half(b0));
      acc1 += a1 * __half2float(__high2half(b1));
    }

    *reinterpret_cast<__half2 *>(&C[m * N + n2]) =
        __floats2half2_rn(acc0, acc1);
  } else if (n2 < N) {
    // 奇数 N 尾部：标量路径。
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
      acc += __half2float(A[m * K + k]) * __half2float(B[k * N + n2]);
    }
    C[m * N + n2] = __float2half(acc);
  }
}

}  // namespace

void gemm_launch_v1(const __half *A, const __half *B, __half *C, int M, int N,
                    int K, cudaStream_t stream) {
  dim3 block(kThreadsX, kThreadsY);
  dim3 grid(ceil_div(ceil_div(N, 2), kThreadsX), ceil_div(M, kThreadsY));
  CUDA_LAUNCH(gemm_v1_kernel, grid, block, 0, stream, A, B, C, M, N, K);
}

}  // namespace cudallm
