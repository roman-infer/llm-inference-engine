// 激活函数、逐元素操作和逐行 softmax。

#include <math.h>

#include "cuda_llm/ops.h"

namespace cudallm {
namespace op {

namespace {

constexpr int kBlock = 256;

#define ELEMENTWISE_LAUNCH(kernel, n, stream, ...)                            \
  do {                                                                        \
    const int64_t blocks = (n + kBlock - 1) / kBlock;                         \
    CUDA_LAUNCH(kernel, static_cast<unsigned>(blocks), kBlock, 0, stream,     \
                __VA_ARGS__, n);                                              \
  } while (0)

__global__ void silu_kernel(const __half *__restrict__ x,
                            __half *__restrict__ y, int64_t n) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    float v = __half2float(x[idx]);
    y[idx] = __float2half(v / (1.0f + expf(-v)));
  }
}

__global__ void gelu_kernel(const __half *__restrict__ x,
                            __half *__restrict__ y, int64_t n) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    float v = __half2float(x[idx]);
    // 精确 GELU：0.5 * v * (1 + erf(v / sqrt(2)))
    y[idx] = __float2half(0.5f * v * (1.0f + erff(v * 0.70710678f)));
  }
}

__global__ void add_kernel(const __half *__restrict__ a,
                           const __half *__restrict__ b,
                           __half *__restrict__ y, int64_t n) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    y[idx] = __float2half(__half2float(a[idx]) + __half2float(b[idx]));
  }
}

__global__ void add_inplace_kernel(__half *__restrict__ a,
                                   const __half *__restrict__ b, int64_t n) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    a[idx] = __float2half(__half2float(a[idx]) + __half2float(b[idx]));
  }
}

__global__ void mul_kernel(const __half *__restrict__ a,
                           const __half *__restrict__ b,
                           __half *__restrict__ y, int64_t n) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    y[idx] = __float2half(__half2float(a[idx]) * __half2float(b[idx]));
  }
}

// 块内归约（通过 smem[0] 使所有线程获得结果）。
__device__ float block_max_v(float v, float *smem) {
  int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  for (int off = 16; off > 0; off >>= 1)
    v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
  if (lane == 0) smem[wid] = v;
  __syncthreads();
  v = (threadIdx.x < kBlock / 32) ? smem[threadIdx.x] : -INFINITY;
  if (wid == 0) {
    for (int off = 16; off > 0; off >>= 1)
      v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
    if (lane == 0) smem[0] = v;
  }
  __syncthreads();
  return smem[0];
}
__device__ float block_sum_v(float v, float *smem) {
  int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  for (int off = 16; off > 0; off >>= 1)
    v += __shfl_down_sync(0xffffffffu, v, off);
  if (lane == 0) smem[wid] = v;
  __syncthreads();
  v = (threadIdx.x < kBlock / 32) ? smem[threadIdx.x] : 0.0f;
  if (wid == 0) {
    for (int off = 16; off > 0; off >>= 1)
      v += __shfl_down_sync(0xffffffffu, v, off);
    if (lane == 0) smem[0] = v;
  }
  __syncthreads();
  return smem[0];
}

__global__ void softmax_kernel(__half *x, int rows, int cols) {
  __shared__ float smem[32];
  const int row = blockIdx.x;
  if (row >= rows) return;
  __half *xr = x + static_cast<int64_t>(row) * cols;

  float mx = -INFINITY;
  for (int c = threadIdx.x; c < cols; c += kBlock)
    mx = fmaxf(mx, __half2float(xr[c]));
  mx = block_max_v(mx, smem);

  float sum = 0.0f;
  for (int c = threadIdx.x; c < cols; c += kBlock) {
    float e = expf(__half2float(xr[c]) - mx);
    sum += e;
    xr[c] = __float2half(e);
  }
  sum = block_sum_v(sum, smem);

  for (int c = threadIdx.x; c < cols; c += kBlock) {
    xr[c] = __float2half(__half2float(xr[c]) / sum);
  }
}

}  // namespace

void silu(const Tensor &x, Tensor &y, cudaStream_t stream) {
  ELEMENTWISE_LAUNCH(silu_kernel, x.numel(), stream, x.data_ptr<__half>(),
                     y.data_ptr<__half>());
}
void gelu(const Tensor &x, Tensor &y, cudaStream_t stream) {
  ELEMENTWISE_LAUNCH(gelu_kernel, x.numel(), stream, x.data_ptr<__half>(),
                     y.data_ptr<__half>());
}
void add(const Tensor &a, const Tensor &b, Tensor &y, cudaStream_t stream) {
  ELEMENTWISE_LAUNCH(add_kernel, a.numel(), stream, a.data_ptr<__half>(),
                     b.data_ptr<__half>(), y.data_ptr<__half>());
}
void add_inplace(Tensor &a, const Tensor &b, cudaStream_t stream) {
  ELEMENTWISE_LAUNCH(add_inplace_kernel, a.numel(), stream,
                     a.data_ptr<__half>(), b.data_ptr<__half>());
}
void mul(const Tensor &a, const Tensor &b, Tensor &y, cudaStream_t stream) {
  ELEMENTWISE_LAUNCH(mul_kernel, a.numel(), stream, a.data_ptr<__half>(),
                     b.data_ptr<__half>(), y.data_ptr<__half>());
}
void softmax(Tensor &x, int rows, int cols, cudaStream_t stream) {
  CUDA_LAUNCH(softmax_kernel, rows, kBlock, 0, stream,
              x.data_ptr<__half>(), rows, cols);
}

}  // namespace op
}  // namespace cudallm
