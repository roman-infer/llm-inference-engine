// LLaMA 风格模型：Pre-RMSNorm、RoPE、MHA/MQA/GQA、SwiGLU MLP、无 bias，
// 可选输出嵌入共享权重。权重预打包策略与 GPT-2 相同。

#include <string>
#include <unordered_map>
#include <utility>

#include "cuda_llm/model.h"
#include "cuda_llm/model_common.h"
#include "cuda_llm/ops.h"

namespace cudallm {

namespace {

using namespace model_common;

struct LlamaLayer {
  Tensor attn_norm;
  LinearW q, k, v, o;
  Tensor mlp_norm;
  LinearW gate, up, down;
};

class LlamaModel : public LanguageModel {
 public:
  LlamaModel(std::unordered_map<std::string, Tensor> weights,
             const ModelConfig &cfg, Context &ctx)
      : cfg_(cfg) {
    tok_emb_ = take(weights, "tok_emb");

    layers_.resize(cfg.n_layer);
    for (int i = 0; i < cfg.n_layer; ++i) {
      LlamaLayer &L = layers_[i];
      const std::string p = "layers" + std::to_string(i) + ".";
      L.attn_norm = take(weights, p + "attn_norm.w");
      L.q = make_linear(weights, p + "q.w", "", ctx);
      L.k = make_linear(weights, p + "k.w", "", ctx);
      L.v = make_linear(weights, p + "v.w", "", ctx);
      L.o = make_linear(weights, p + "o.w", "", ctx);
      L.mlp_norm = take(weights, p + "mlp_norm.w");
      L.gate = make_linear(weights, p + "gate.w", "", ctx);
      L.up = make_linear(weights, p + "up.w", "", ctx);
      L.down = make_linear(weights, p + "down.w", "", ctx);
    }
    final_norm_ = take(weights, "norm.w");

    // lm_head：独立权重，否则与嵌入层共享。
    auto it = weights.find("lm_head");
    if (it != weights.end()) lm_head_ = std::move(it->second);

    const int d = cfg.n_embd / cfg.n_head;
    k_cache_ = Tensor({cfg.n_layer, cfg.n_kv_head, cfg.max_seq_len, d},
                      DType::kFloat16, ctx.device(), "k_cache");
    v_cache_ = Tensor({cfg.n_layer, cfg.n_kv_head, cfg.max_seq_len, d},
                      DType::kFloat16, ctx.device(), "v_cache");
    reset_caches(ctx.stream());

    scratch_ = Tensor(
        {static_cast<int64_t>(cfg.n_head) * cfg.max_seq_len *
         cfg.max_seq_len},
        DType::kFloat16, ctx.device(), "attn_scratch");
    logits_ = Tensor({cfg.vocab_size}, DType::kFloat16, ctx.device(),
                     "logits");
  }

  const ModelConfig &config() const override { return cfg_; }

  void reset_caches(cudaStream_t stream) override {
    k_cache_.zero_async(stream);
    v_cache_.zero_async(stream);
  }

  const Tensor &forward(const Tensor &ids, int n, int past_len,
                        Context &ctx) override {
    cudaStream_t s = ctx.stream();
    const int E = cfg_.n_embd;
    const int H = cfg_.n_head;
    const int KH = cfg_.n_kv_head;
    const int d = E / H;
    const int I = cfg_.n_inter;
    const int kv_len = past_len + n;

    Tensor x({n, E}, DType::kFloat16, ctx.device(), "x");
    op::embedding(tok_emb_, ids, x, s);

    for (int li = 0; li < cfg_.n_layer; ++li) {
      LlamaLayer &L = layers_[li];

      // --- 注意力 ---
      Tensor n1({n, E}, DType::kFloat16, ctx.device());
      op::rms_norm(x, L.attn_norm, n1, cfg_.eps, s);

      Tensor q({n, H * d}, DType::kFloat16, ctx.device());
      Tensor k({n, KH * d}, DType::kFloat16, ctx.device());
      Tensor v({n, KH * d}, DType::kFloat16, ctx.device());
      op::linear_gemm(n1, for_gemm(L.q), q, GemmVersion::kV2Tiled, s);
      op::linear_gemm(n1, for_gemm(L.k), k, GemmVersion::kV2Tiled, s);
      op::linear_gemm(n1, for_gemm(L.v), v, GemmVersion::kV2Tiled, s);

      Tensor qv = q.view({n, H, d});
      Tensor kv = k.view({n, KH, d});
      Tensor vv = v.view({n, KH, d});
      op::rope_inplace(qv, kv, past_len, cfg_.theta, s);

      Tensor kc = k_cache_.slice(0, li, li + 1)
                      .view({KH, cfg_.max_seq_len, d});
      Tensor vc = v_cache_.slice(0, li, li + 1)
                      .view({KH, cfg_.max_seq_len, d});
      op::kv_append(kc, kv, past_len, s);
      op::kv_append(vc, vv, past_len, s);

      Tensor ao({n, H, d}, DType::kFloat16, ctx.device());
      Tensor sc = Tensor::wrap(scratch_.data_ptr(),
                               {static_cast<int64_t>(H) * n * kv_len},
                               DType::kFloat16, ctx.device(), "attn_scratch");
      op::attention(qv, kc, vc, ao, H, KH, past_len, true, sc, s);

      Tensor aof = ao.view({n, E});
      Tensor p1({n, E}, DType::kFloat16, ctx.device());
      op::linear_gemm(aof, for_gemm(L.o), p1, GemmVersion::kV2Tiled, s);

      Tensor x2({n, E}, DType::kFloat16, ctx.device());
      op::add(x, p1, x2, s);
      x = std::move(x2);

      // --- SwiGLU MLP ---
      Tensor n2({n, E}, DType::kFloat16, ctx.device());
      op::rms_norm(x, L.mlp_norm, n2, cfg_.eps, s);

      Tensor gate({n, I}, DType::kFloat16, ctx.device());
      Tensor up({n, I}, DType::kFloat16, ctx.device());
      op::linear_gemm(n2, for_gemm(L.gate), gate, GemmVersion::kV2Tiled, s);
      op::linear_gemm(n2, for_gemm(L.up), up, GemmVersion::kV2Tiled, s);

      Tensor sg({n, I}, DType::kFloat16, ctx.device());
      op::silu(gate, sg, s);
      Tensor fu({n, I}, DType::kFloat16, ctx.device());
      op::mul(sg, up, fu, s);

      Tensor p2({n, E}, DType::kFloat16, ctx.device());
      op::linear_gemm(fu, for_gemm(L.down), p2, GemmVersion::kV2Tiled, s);

      Tensor x3({n, E}, DType::kFloat16, ctx.device());
      op::add(x, p2, x3, s);
      x = std::move(x3);
    }

    Tensor xf({n, E}, DType::kFloat16, ctx.device());
    op::rms_norm(x, final_norm_, xf, cfg_.eps, s);

    Tensor last = xf.slice(0, n - 1, n).view({1, E});
    const Tensor &head = lm_head_.data_ptr() ? lm_head_ : tok_emb_;
    op::linear_gemv(last, {nullptr, &head, nullptr}, logits_, s);
    return logits_;
  }

 private:
  ModelConfig cfg_;
  Tensor tok_emb_, lm_head_;
  std::vector<LlamaLayer> layers_;
  Tensor final_norm_;
  Tensor k_cache_, v_cache_, scratch_, logits_;
};

}  // namespace

std::unique_ptr<LanguageModel> build_llama(
    std::unordered_map<std::string, Tensor> weights, const ModelConfig &cfg,
    Context &ctx) {
  return std::make_unique<LlamaModel>(std::move(weights), cfg, ctx);
}

}  // namespace cudallm
