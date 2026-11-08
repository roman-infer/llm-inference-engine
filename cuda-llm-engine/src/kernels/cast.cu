// 精度转换 kernel：F32 <-> F16，外加用于将线性层权重预打包为
// GEMM 布局的二维转置。

#include "cuda_llm/ops.h"

namespace cudallm {
namespace op {

namespace {

constexpr int kBlock = 256;

__global__ void f32_to_f16_kernel(const float *__restrict__ src,
                                  __half *__restrict__ dst, int64_t n) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) dst[idx] = __float2half(src[idx]);
}

__global__ void f16_to_f32_kernel(const __half *__restrict__ src,
                                  float *__restrict__ dst, int64_t n) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) dst[idx] = __half2float(src[idx]);
}

// src [r,c] -> dst [c,r]，分块向量化加载，通过共享内存中转转置，
// 保证读写都是合并访存。
constexpr int kTile = 32;
__global__ void transpose_2d_kernel(const __half *__restrict__ src,
                                    __half *__restrict__ dst, int r, int c) {
  __shared__ __half tile[kTile][kTile + 1];

  const int x = blockIdx.x * kTile + threadIdx.x;  // 列
  const int y = blockIdx.y * kTile + threadIdx.y;  // 行
  if (x < c && y < r) {
    tile[threadIdx.y][threadIdx.x] =
        src[static_cast<int64_t>(y) * c + x];
  }
  __syncthreads();

  // 转置后的块：dst 维度为 [c,r]；块索引互换。
  const int x2 = blockIdx.y * kTile + threadIdx.x;  // dst 的列 -> 旧行
  const int y2 = blockIdx.x * kTile + threadIdx.y;  // dst 的行 -> 旧列
  if (x2 < r && y2 < c) {
    dst[static_cast<int64_t>(y2) * r + x2] =
        tile[threadIdx.x][threadIdx.y];
  }
}

}  // namespace

void cast_f32_to_f16(const float *src, __half *dst, int64_t n,
                     cudaStream_t stream) {
  const int64_t blocks = (n + kBlock - 1) / kBlock;
  CUDA_LAUNCH(f32_to_f16_kernel, static_cast<unsigned>(blocks), kBlock, 0,
              stream, src, dst, n);
}

void cast_f16_to_f32(const __half *src, float *dst, int64_t n,
                     cudaStream_t stream) {
  const int64_t blocks = (n + kBlock - 1) / kBlock;
  CUDA_LAUNCH(f16_to_f32_kernel, static_cast<unsigned>(blocks), kBlock, 0,
              stream, src, dst, n);
}

void transpose_2d(const Tensor &src, Tensor &dst, cudaStream_t stream) {
  const int r = static_cast<int>(src.size(0));
  const int c = static_cast<int>(src.size(1));
  dim3 block(32, 8);
  dim3 grid(ceil_div(c, kTile), ceil_div(r, kTile));
  CUDA_LAUNCH(transpose_2d_kernel, grid, block, 0, stream,
              src.data_ptr<__half>(), dst.data_ptr<__half>(), r, c);
}

}  // namespace op
}  // namespace cudallm
