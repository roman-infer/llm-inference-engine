// GEMM v4：decode 阶段 GEMV，权重布局对 KV cache 友好。
//
// 自回归解码时输入是单个 token（M=1），因此 GEMM 退化为矩阵-向量
// 乘法，纯受带宽限制。权重以 w[N,K]（输出 x 输入）存储：每个输出行
// 是 HBM 中连续的 K 个元素流。
//
//   y[n] = sum_k x[k] * w[n][k]
//
// 一个 warp 负责一个输出 n：lane 以 64 为步长沿 K 遍历（half2 对），
// 然后通过 warp shuffle 将 32 个部分和合并。N 个 warp 并发运行，
// 从而饱和内存带宽。生成的 token 投影在 decode 循环中送入
// KV cache 追加操作（见 kvcache.cu）。

#include "cuda_llm/gemm.h"

namespace cudallm {

namespace {

__global__ void gemv_kernel(const __half *__restrict__ x,
                            const __half *__restrict__ w,
                            __half *__restrict__ y, int K, int N) {
  const int n = blockIdx.x;
  const int lane = threadIdx.x;
  const __half *row = w + n * K;

  float acc = 0.0f;
  int k = lane * 2;
  for (; k + 1 < K; k += 64) {
    const __half2 w2 =
        *reinterpret_cast<const __half2 *>(row + k);
    const __half2 x2 =
        *reinterpret_cast<const __half2 *>(x + k);
    const __half2 p = __hmul2(w2, x2);
    acc += __half2float(__low2half(p)) + __half2float(__high2half(p));
  }
  if (k < K) {  // 奇数 K 尾部，最多一个 lane 处理
    acc += __half2float(row[k]) * __half2float(x[k]);
  }

  for (int off = 16; off > 0; off >>= 1) {
    acc += __shfl_down_sync(0xffffffffu, acc, off);
  }
  if (lane == 0) y[n] = __float2half(acc);
}

}  // namespace

void gemv_launch(const __half *x, const __half *w, __half *y, int K, int N,
                 cudaStream_t stream) {
  dim3 grid(N);
  dim3 block(32);
  CUDA_LAUNCH(gemv_kernel, grid, block, 0, stream, x, w, y, K, N);
}

}  // namespace cudallm
