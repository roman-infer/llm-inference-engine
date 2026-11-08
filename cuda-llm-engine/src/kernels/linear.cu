// 线性层封装：prefill 阶段使用转置权重的 GEMM，
// decode 阶段使用 [out,in] 权重的 GEMV，外加 bias 广播。

#include "cuda_llm/ops.h"

namespace cudallm {
namespace op {

namespace {

constexpr int kBlock = 256;

// y[m,c] += bias[c]，grid-stride 遍历。
__global__ void bias_add_kernel(__half *__restrict__ y,
                                const __half *__restrict__ bias, int rows,
                                int cols) {
  const int64_t total = static_cast<int64_t>(rows) * cols;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
       idx < total; idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int c = static_cast<int>(idx % cols);
    y[idx] = __float2half(__half2float(y[idx]) + __half2float(bias[c]));
  }
}

void add_bias(Tensor &y, const Tensor &bias, int rows, int cols,
              cudaStream_t stream) {
  const int64_t total = static_cast<int64_t>(rows) * cols;
  const int blocks = static_cast<int>(
      std::min<int64_t>(ceil_div(total, kBlock), 2048));
  CUDA_LAUNCH(bias_add_kernel, blocks, kBlock, 0, stream,
              y.data_ptr<__half>(), bias.data_ptr<__half>(), rows, cols);
}

}  // namespace

void linear_gemm(const Tensor &x, const LinearParams &lp, Tensor &y,
                 GemmVersion version, cudaStream_t stream) {
  const int k = static_cast<int>(x.size(-1));
  const int m = static_cast<int>(x.numel() / k);
  const int n = static_cast<int>(lp.wt->size(1));

  gemm_launch(version, x.data_ptr<__half>(), lp.wt->data_ptr<__half>(),
              y.data_ptr<__half>(), m, n, k, stream);
  if (lp.bias) add_bias(y, *lp.bias, m, n, stream);
}

void linear_gemv(const Tensor &x, const LinearParams &lp, Tensor &y,
                 cudaStream_t stream) {
  const int k = static_cast<int>(x.size(-1));
  const int m = static_cast<int>(x.numel() / k);
  const int n = static_cast<int>(lp.w->size(0));
  if (m != 1) {
    throw std::runtime_error("linear_gemv: expected a single token (M=1)");
  }

  gemv_launch(x.data_ptr<__half>(), lp.w->data_ptr<__half>(),
              y.data_ptr<__half>(), k, n, stream);
  if (lp.bias) add_bias(y, *lp.bias, 1, n, stream);
}

namespace {

// src [rows, 3*cols] -> a/b/c [rows, cols]，按列切成三段连续缓冲。
__global__ void split3_kernel(const __half *__restrict__ src,
                              __half *__restrict__ a, __half *__restrict__ b,
                              __half *__restrict__ c, int rows, int cols) {
  const int64_t total = static_cast<int64_t>(rows) * cols;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
       idx < total; idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t r = idx / cols;
    const int64_t col = idx - r * cols;
    const __half *p = src + (r * 3 * cols + col);
    a[idx] = p[0];
    b[idx] = p[cols];
    c[idx] = p[2 * cols];
  }
}

}  // namespace

void split_qkv(const Tensor &src, Tensor &q, Tensor &k, Tensor &v,
               cudaStream_t stream) {
  const int rows = static_cast<int>(src.size(0));
  const int cols = static_cast<int>(src.size(1) / 3);
  const int64_t total = static_cast<int64_t>(rows) * cols;
  const int blocks = static_cast<int>(
      std::min<int64_t>(ceil_div(total, kBlock), 2048));
  CUDA_LAUNCH(split3_kernel, blocks, kBlock, 0, stream,
              src.data_ptr<__half>(), q.data_ptr<__half>(),
              k.data_ptr<__half>(), v.data_ptr<__half>(), rows, cols);
}

}  // namespace op
}  // namespace cudallm
