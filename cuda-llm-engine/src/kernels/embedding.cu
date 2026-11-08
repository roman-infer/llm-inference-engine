// 嵌入查表：out[s,:] = table[ids[s], :]。

#include <algorithm>

#include "cuda_llm/ops.h"

namespace cudallm {
namespace op {

namespace {

constexpr int kBlock = 256;

__global__ void embedding_kernel(const __half *__restrict__ table,
                                 const int *__restrict__ ids,
                                 __half *__restrict__ out, int seq, int dim) {
  const int64_t total = static_cast<int64_t>(seq) * dim;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
       idx < total; idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int d = static_cast<int>(idx % dim);
    const int s = static_cast<int>(idx / dim);
    const int id = ids[s];
    out[idx] = table[static_cast<int64_t>(id) * dim + d];
  }
}

}  // namespace

void embedding(const Tensor &table, const Tensor &ids, Tensor &out,
               cudaStream_t stream) {
  const int seq = static_cast<int>(ids.numel());
  const int dim = static_cast<int>(table.size(1));
  const int64_t total = static_cast<int64_t>(seq) * dim;
  const int blocks = static_cast<int>(
      std::min<int64_t>(ceil_div(total, kBlock), 2048));
  CUDA_LAUNCH(embedding_kernel, blocks, kBlock, 0, stream,
              table.data_ptr<__half>(), ids.data_ptr<int>(),
              out.data_ptr<__half>(), seq, dim);
}

}  // namespace op
}  // namespace cudallm
