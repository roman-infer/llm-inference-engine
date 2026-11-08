// 旋转位置编码（RoPE），原地修改。
//
// x 布局：[seq, heads, head_dim]。对于绝对位置 p 处的每对 (x[2i], x[2i+1])：
//   angle = p * theta_base^(-2i/head_dim)
//   x0' = x0 cos - x1 sin
//   x1' = x0 sin + x1 cos
// q 和 k 使用同一个 kernel，各自使用对应的 head 数。

#include <algorithm>

#include "cuda_llm/ops.h"

#include <math.h>

namespace cudallm {
namespace op {

namespace {

constexpr int kBlock = 256;

__global__ void rope_kernel(__half *x, int seq, int heads, int head_dim,
                            int pos0, float theta_base) {
  const int pairs = head_dim / 2;
  const int64_t total = static_cast<int64_t>(seq) * heads * pairs;

  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
       idx < total; idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int d2 = static_cast<int>(idx % pairs);
    const int h = static_cast<int>((idx / pairs) % heads);
    const int s = static_cast<int>(idx / (static_cast<int64_t>(pairs) *
                                          heads));
    const int p = pos0 + s;

    const float angle =
        p * powf(theta_base, -static_cast<float>(2 * d2) / head_dim);
    const float cs = cosf(angle);
    const float sn = sinf(angle);

    __half *el =
        &x[(static_cast<int64_t>(s) * heads + h) * head_dim + 2 * d2];
    const float x0 = __half2float(el[0]);
    const float x1 = __half2float(el[1]);
    el[0] = __float2half(x0 * cs - x1 * sn);
    el[1] = __float2half(x0 * sn + x1 * cs);
  }
}

void launch_rope(Tensor &x, int heads, int pos0, float theta_base,
                 cudaStream_t stream) {
  const int seq = static_cast<int>(x.size(0));
  const int head_dim = static_cast<int>(x.size(2));
  const int pairs = head_dim / 2;
  const int64_t total = static_cast<int64_t>(seq) * heads * pairs;
  const int blocks = static_cast<int>(
      std::min<int64_t>(ceil_div(total, kBlock), 2048));
  CUDA_LAUNCH(rope_kernel, blocks, kBlock, 0, stream,
              x.data_ptr<__half>(), seq, heads, head_dim, pos0, theta_base);
}

}  // namespace

void rope_inplace(Tensor &q, Tensor &k, int pos0, float theta_base,
                  cudaStream_t stream) {
  const int n_heads = static_cast<int>(q.size(1));
  const int n_kv_heads = static_cast<int>(k.size(1));
  launch_rope(q, n_heads, pos0, theta_base, stream);
  launch_rope(k, n_kv_heads, pos0, theta_base, stream);
}

}  // namespace op
}  // namespace cudallm
