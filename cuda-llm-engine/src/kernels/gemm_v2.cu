// GEMM v2：共享内存分块 + 寄存器分块（register tiling）。
//
// 相对 v1 的两处改动：
//   1. 协作搬运：一个 block 负责 BM×BN = 64×64 的输出块，A/B 的分块
//      先搬进共享内存，块内所有线程复用，全局访存从 O(BM·BN·BK) 降到
//      O(BM·BK + BK·BN)。
//   2. 寄存器分块：每个线程不再只算 1 个输出，而是算 TM×TN = 4×4 = 16 个。
//      每读一次共享内存的 A、B 分量，能摊到 4 次 FMA 上，
//      算术强度从 1 FMA / 2 次 smem 读提升到 4 FMA / 1 次 smem 读。
//
// 布局与 bank conflict：
//   As 按 [BK][BM + PAD] 存放（行内沿 m 连续），PAD = 8；
//   BM+PAD = 72 ≡ 8 (mod 32)，16 个 ty 落在 8 个 bank 上各 2 个 → 2 路冲突，
//   远好于不 padding 时的 16 路。
//
// 线程映射：blockDim = (16,16)，tx 管 n、ty 管 m。
//
// 注意：BN=64 对 N 很小（如 decode 的 N=8）是浪费的——那种形状该走 v4 GEMV，
//   见 README 的“按形状特化”一节。

#include "cuda_llm/gemm.h"

namespace cudallm {

namespace {

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;
constexpr int TM = 4;
constexpr int TN = 4;
constexpr int PAD = 8;
constexpr int THREADS = (BM / TM) * (BN / TN);  // 16 * 16 = 256

__global__ void gemm_v2_kernel(const __half *__restrict__ A,
                               const __half *__restrict__ B,
                               __half *__restrict__ C, int M, int N, int K) {
  __shared__ __half As[BK][BM + PAD];
  __shared__ __half Bs[BK][BN + PAD];

  const int tx = threadIdx.x;  // 0..15，负责 n
  const int ty = threadIdx.y;  // 0..15，负责 m
  const int tid = ty * (BN / TN) + tx;

  const int bm = blockIdx.y * BM;
  const int bn = blockIdx.x * BN;

  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

  for (int bk = 0; bk < K; bk += BK) {
    // A 分块：BM×BK = 1024 个元素，256 个线程每个搬 4 个。
    // 线程 i 取 (m, k) = (i / BK, i % BK)，沿 k 连续 → 全局合并访存。
#pragma unroll
    for (int i = tid; i < BM * BK; i += THREADS) {
      const int m = i / BK;
      const int k = i - m * BK;
      const int gm = bm + m;
      const int gk = bk + k;
      As[k][m] = (gm < M && gk < K) ? A[gm * K + gk] : __float2half(0.0f);
    }
    // B 分块：BK×BN = 1024 个元素，沿 n 连续 → 全局合并访存。
#pragma unroll
    for (int i = tid; i < BK * BN; i += THREADS) {
      const int k = i / BN;
      const int n = i - k * BN;
      const int gk = bk + k;
      const int gn = bn + n;
      Bs[k][n] = (gk < K && gn < N) ? B[gk * N + gn] : __float2half(0.0f);
    }
    __syncthreads();

    const int rm = ty * TM;
    const int rn = tx * TN;
#pragma unroll
    for (int k = 0; k < BK; ++k) {
      __half a[TM];
      __half b[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i) a[i] = As[k][rm + i];
#pragma unroll
      for (int j = 0; j < TN; ++j) b[j] = Bs[k][rn + j];
#pragma unroll
      for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          acc[i][j] += __half2float(a[i]) * __half2float(b[j]);
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int gm = bm + ty * TM + i;
    if (gm >= M) continue;
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int gn = bn + tx * TN + j;
      if (gn < N) C[gm * N + gn] = __float2half(acc[i][j]);
    }
  }
}

}  // namespace

void gemm_launch_v2(const __half *A, const __half *B, __half *C, int M, int N,
                    int K, cudaStream_t stream) {
  dim3 block(BN / TN, BM / TM);
  dim3 grid(ceil_div(N, BN), ceil_div(M, BM));
  CUDA_LAUNCH(gemm_v2_kernel, grid, block, 0, stream, A, B, C, M, N, K);
}

}  // namespace cudallm
