#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "cuda_llm/context.h"
#include "cuda_llm/tensor.h"

namespace cudallm {

struct ModelConfig {
  int n_layer = 12;
  int n_head = 12;
  int n_kv_head = 12;  // < n_head 时为 GQA；为 1 时为 MQA
  int n_embd = 768;
  int n_inter = 3072;
  int max_seq_len = 1024;
  int vocab_size = 50257;
  float eps = 1e-5f;
  float theta = 10000.0f;
  DType weight_dtype = DType::kFloat16;
};

// 已加载的语言模型持有其权重、KV cache 和临时工作区。
class LanguageModel {
 public:
  virtual ~LanguageModel() = default;

  // 运行 token id 序列 [n_tokens]（int32，设备端）。仅计算最后一个
  // token 的 logits；返回的张量归模型所有，在下一次 forward()
  // 调用之前有效。
  virtual const Tensor &forward(const Tensor &ids, int n_tokens,
                                int past_len, Context &ctx) = 0;

  virtual const ModelConfig &config() const = 0;

  // 为新序列清零/重置 KV cache。
  virtual void reset_caches(cudaStream_t stream) = 0;
};

// 从 name->Tensor map 构建模型（原始 HF 布局的 FP16 权重）。
std::unique_ptr<LanguageModel> build_gpt2(
    std::unordered_map<std::string, Tensor> weights, const ModelConfig &cfg,
    Context &ctx);

std::unique_ptr<LanguageModel> build_llama(
    std::unordered_map<std::string, Tensor> weights, const ModelConfig &cfg,
    Context &ctx);

}  // namespace cudallm
