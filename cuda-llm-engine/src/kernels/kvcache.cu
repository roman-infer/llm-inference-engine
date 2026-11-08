// KV cache 追加。
//
// cache : [n_kv_heads, max_seq, head_dim]
// src   : [seq, n_kv_heads, head_dim]
// 将 src 拷贝到 cache 中绝对位置 [pos0, pos0+seq) 处。
// 同时用于 prefill（一次性处理整个 prompt）和 decode（每步一个 token）。

#include <algorithm>

#include "cuda_llm/ops.h"

namespace cudallm {
namespace op {

namespace {

constexpr int kBlock = 256;

__global__ void kv_append_kernel(__half *__restrict__ cache,
                                 const __half *__restrict__ src, int seq,
                                 int heads, int max_seq, int head_dim,
                                 int pos0) {
  const int64_t total =
      static_cast<int64_t>(seq) * heads * head_dim;

  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
       idx < total; idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int d = static_cast<int>(idx % head_dim);
    const int h = static_cast<int>((idx / head_dim) % heads);
    const int s = static_cast<int>(idx /
                                   (static_cast<int64_t>(head_dim) * heads));
    const int t = pos0 + s;

    cache[(static_cast<int64_t>(h) * max_seq + t) * head_dim + d] =
        src[idx];
  }
}

}  // namespace

void kv_append(Tensor &cache, const Tensor &src, int pos0,
               cudaStream_t stream) {
  const int seq = static_cast<int>(src.size(0));
  const int heads = static_cast<int>(src.size(1));
  const int head_dim = static_cast<int>(src.size(2));
  const int max_seq = static_cast<int>(cache.size(1));
  const int64_t total =
      static_cast<int64_t>(seq) * heads * head_dim;
  const int blocks = static_cast<int>(
      std::min<int64_t>(ceil_div(total, kBlock), 2048));

  CUDA_LAUNCH(kv_append_kernel, blocks, kBlock, 0, stream,
              cache.data_ptr<__half>(), src.data_ptr<__half>(), seq, heads,
              max_seq, head_dim, pos0);
}

}  // namespace op
}  // namespace cudallm
