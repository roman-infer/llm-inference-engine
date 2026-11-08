#pragma once

// GPT-2 和 LLaMA 模型构建的共享辅助函数：权重打包（LinearW）、
// 查找（take）以及 GEMM/GEMV 参数适配器。以前每个模型文件都重复定义
// 这些内容；此头文件是唯一的事实来源。

#include <string>
#include <unordered_map>
#include <utility>

#include "cuda_llm/context.h"
#include "cuda_llm/ops.h"
#include "cuda_llm/tensor.h"

namespace cudallm {
namespace model_common {

// 同时持有两种布局的线性层：w[out,in] 用于 decode GEMV，
// wt[in,out] 用于 prefill GEMM，外加可选的 bias（无时为空）。
struct LinearW {
  Tensor w;
  Tensor wt;
  Tensor bias;
};

inline op::LinearParams for_gemm(const LinearW &lw) {
  return {&lw.wt, nullptr, lw.bias.data_ptr() ? &lw.bias : nullptr};
}

// 加载权重张量，将其转置为 GEMM 布局，并可选地获取 bias。
// 该层没有 bias 时传入空的 bkey。
inline LinearW make_linear(std::unordered_map<std::string, Tensor> &weights,
                            const std::string &wkey, const std::string &bkey,
                            Context &ctx) {
  LinearW lw;
  lw.w = std::move(weights.at(wkey));
  const int out = static_cast<int>(lw.w.size(0));
  const int in = static_cast<int>(lw.w.size(1));
  lw.wt = Tensor({in, out}, DType::kFloat16, ctx.device(), wkey + ".t");
  op::transpose_2d(lw.w, lw.wt, ctx.stream());
  if (!bkey.empty()) {
    auto bit = weights.find(bkey);
    if (bit != weights.end()) lw.bias = std::move(bit->second);
  }
  return lw;
}

inline Tensor take(std::unordered_map<std::string, Tensor> &weights,
                   const std::string &key) {
  return std::move(weights.at(key));
}

// 同一件事的另一半：HuggingFace GPT-2 的 Conv1D 权重按 [in, out] 存储，
// 与 torch.nn.Linear 的 [out, in] 恰好相反。直接套用 make_linear 会把
// 维度读反（qkv 只算出 768 个输出而不是 2304 个）。
// 这里把存储布局直接当作 GEMM 需要的 wt[K, N]，再转置出 GEMV 用的 w[N, K]。
inline LinearW make_linear_conv1d(
    std::unordered_map<std::string, Tensor> &weights, const std::string &wkey,
    const std::string &bkey, Context &ctx) {
  LinearW lw;
  lw.wt = std::move(weights.at(wkey));
  const int in = static_cast<int>(lw.wt.size(0));
  const int out = static_cast<int>(lw.wt.size(1));
  lw.w = Tensor({out, in}, DType::kFloat16, ctx.device(), wkey + ".w");
  op::transpose_2d(lw.wt, lw.w, ctx.stream());
  if (!bkey.empty()) {
    auto bit = weights.find(bkey);
    if (bit != weights.end()) lw.bias = std::move(bit->second);
  }
  return lw;
}

}  // namespace model_common
}  // namespace cudallm
