// LayerNorm / RMSNorm kernel。每行一个 block，通过 warp shuffle
// 加一个小共享缓冲区完成整块归约。

#include "cuda_llm/ops.h"

namespace cudallm {
namespace op {

namespace {

constexpr int kBlock = 256;

// 整块求和归约；结果在每个线程上都可用（通过 smem[0] 返回）。
__device__ float block_reduce_total(float v, float *smem) {
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;

  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_down_sync(0xffffffffu, v, off);
  }
  if (lane == 0) smem[wid] = v;
  __syncthreads();

  v = (threadIdx.x < kBlock / 32) ? smem[threadIdx.x] : 0.0f;
  if (wid == 0) {
    for (int off = 16; off > 0; off >>= 1) {
      v += __shfl_down_sync(0xffffffffu, v, off);
    }
    if (lane == 0) smem[0] = v;
  }
  __syncthreads();
  return smem[0];
}

__global__ void rms_norm_kernel(const __half *__restrict__ x,
                                const __half *__restrict__ weight,
                                __half *__restrict__ y, int rows, int cols,
                                float eps) {
  __shared__ float smem[32];
  const int row = blockIdx.x;
  if (row >= rows) return;

  const __half *xr = x + static_cast<int64_t>(row) * cols;
  __half *yr = y + static_cast<int64_t>(row) * cols;

  float sq = 0.0f;
  for (int c = threadIdx.x; c < cols; c += kBlock) {
    float v = __half2float(xr[c]);
    sq += v * v;
  }
  const float mean_sq = block_reduce_total(sq, smem) / cols;
  const float scale = rsqrtf(mean_sq + eps);

  for (int c = threadIdx.x; c < cols; c += kBlock) {
    float v = __half2float(xr[c]) * scale * __half2float(weight[c]);
    yr[c] = __float2half(v);
  }
}

__global__ void layer_norm_kernel(const __half *__restrict__ x,
                                  const __half *__restrict__ weight,
                                  const __half *__restrict__ bias,
                                  __half *__restrict__ y, int rows, int cols,
                                  float eps) {
  __shared__ float smem[32];
  const int row = blockIdx.x;
  if (row >= rows) return;

  const __half *xr = x + static_cast<int64_t>(row) * cols;
  __half *yr = y + static_cast<int64_t>(row) * cols;

  float sum = 0.0f;
  float sq = 0.0f;
  for (int c = threadIdx.x; c < cols; c += kBlock) {
    float v = __half2float(xr[c]);
    sum += v;
    sq += v * v;
  }
  const float mean = block_reduce_total(sum, smem) / cols;
  const float var = block_reduce_total(sq, smem) / cols - mean * mean;
  const float scale = rsqrtf(var + eps);

  for (int c = threadIdx.x; c < cols; c += kBlock) {
    float v = (__half2float(xr[c]) - mean) * scale *
              __half2float(weight[c]);
    if (bias) v += __half2float(bias[c]);
    yr[c] = __float2half(v);
  }
}

}  // namespace

void rms_norm(const Tensor &x, const Tensor &weight, Tensor &y, float eps,
              cudaStream_t stream) {
  const int cols = static_cast<int>(x.size(-1));
  const int rows = static_cast<int>(x.numel() / cols);
  CUDA_LAUNCH(rms_norm_kernel, rows, kBlock, 0, stream,
              x.data_ptr<__half>(), weight.data_ptr<__half>(),
              y.data_ptr<__half>(), rows, cols, eps);
}

void layer_norm(const Tensor &x, const Tensor &weight, const Tensor *bias,
                Tensor &y, float eps, cudaStream_t stream) {
  const int cols = static_cast<int>(x.size(-1));
  const int rows = static_cast<int>(x.numel() / cols);
  const __half *bias_ptr =
      bias ? bias->data_ptr<__half>() : nullptr;
  CUDA_LAUNCH(layer_norm_kernel, rows, kBlock, 0, stream,
              x.data_ptr<__half>(), weight.data_ptr<__half>(), bias_ptr,
              y.data_ptr<__half>(), rows, cols, eps);
}

}  // namespace op
}  // namespace cudallm
