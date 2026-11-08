// GPT-2 模型（124M：12 层，768 隐藏维度，12 头，3072 MLP，FP16）。
//
// 可学习位置嵌入（wpe），Pre-LN，精确 GELU，MHA，输入/输出嵌入共享权重。
// 构建时每个 Linear 权重都转置为 GEMM 布局 wt[in,out]；原始 w[out,in]
// 保留用于 decode GEMV。KV cache 形状为 [n_layer, n_head, max_seq, head_dim]。

#include <string>
#include <unordered_map>
#include <utility>

#include "cuda_llm/model.h"
#include "cuda_llm/model_common.h"
#include "cuda_llm/ops.h"

namespace cudallm {

namespace {

using namespace model_common;

__global__ void arange_kernel(int *out, int n, int start) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = start + i;
}

struct GptLayer {
  Tensor ln1_w, ln1_b;
  LinearW qkv, proj;
  Tensor ln2_w, ln2_b;
  LinearW fc, fcproj;
};

class GPT2Model : public LanguageModel {
 public:
  GPT2Model(std::unordered_map<std::string, Tensor> weights,
            const ModelConfig &cfg, Context &ctx)
      : cfg_(cfg) {
    wte_ = take(weights, "wte");
    wpe_ = take(weights, "wpe");

    layers_.resize(cfg.n_layer);
    for (int i = 0; i < cfg.n_layer; ++i) {
      GptLayer &L = layers_[i];
      const std::string p = "h" + std::to_string(i) + ".";
      L.ln1_w = take(weights, p + "ln1.w");
      L.ln1_b = take(weights, p + "ln1.b");
      L.qkv = make_linear_conv1d(weights, p + "attn.qkv.w",
                                 p + "attn.qkv.b", ctx);
      L.proj = make_linear_conv1d(weights, p + "attn.proj.w",
                                  p + "attn.proj.b", ctx);
      L.ln2_w = take(weights, p + "ln2.w");
      L.ln2_b = take(weights, p + "ln2.b");
      L.fc = make_linear_conv1d(weights, p + "mlp.fc.w",
                                p + "mlp.fc.b", ctx);
      L.fcproj = make_linear_conv1d(weights, p + "mlp.proj.w",
                                    p + "mlp.proj.b", ctx);
    }
    lnf_w_ = take(weights, "lnf.w");
    lnf_b_ = take(weights, "lnf.b");

    const int d = cfg.n_embd / cfg.n_head;
    k_cache_ = Tensor({cfg.n_layer, cfg.n_head, cfg.max_seq_len, d},
                      DType::kFloat16, ctx.device(), "k_cache");
    v_cache_ = Tensor({cfg.n_layer, cfg.n_head, cfg.max_seq_len, d},
                      DType::kFloat16, ctx.device(), "v_cache");
    reset_caches(ctx.stream());

    // 最坏情况下的 attention 分数：n_head * max_seq * max_seq 个 half。
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
    const int d = E / H;
    const int I = cfg_.n_inter;
    const int kv_len = past_len + n;

    // 位置编码。
    Tensor pos({n}, DType::kInt32, ctx.device(), "pos");
    CUDA_LAUNCH(arange_kernel, ceil_div(n, 256), 256, 0, s,
                pos.data_ptr<int>(), n, past_len);

    // Token + 位置嵌入。
    Tensor x({n, E}, DType::kFloat16, ctx.device(), "x");
    op::embedding(wte_, ids, x, s);
    Tensor pe({n, E}, DType::kFloat16, ctx.device(), "pe");
    op::embedding(wpe_, pos, pe, s);
    op::add_inplace(x, pe, s);

    for (int li = 0; li < cfg_.n_layer; ++li) {
      GptLayer &L = layers_[li];

      // --- 注意力 ---
      Tensor n1({n, E}, DType::kFloat16, ctx.device());
      op::layer_norm(x, L.ln1_w, &L.ln1_b, n1, cfg_.eps, s);

      Tensor qkv({n, 3 * E}, DType::kFloat16, ctx.device());
      op::linear_gemm(n1, for_gemm(L.qkv), qkv, GemmVersion::kV2Tiled, s);

      // q/k/v 从融合投影结果中按列切出。Tensor 不带 stride，
      // 对 [n, 3E] 做 slice(1, ...) 再 view 会按连续布局误读，
      // 因此显式拷贝到三个连续缓冲。
      Tensor qc({n, E}, DType::kFloat16, ctx.device());
      Tensor kc2({n, E}, DType::kFloat16, ctx.device());
      Tensor vc2({n, E}, DType::kFloat16, ctx.device());
      op::split_qkv(qkv, qc, kc2, vc2, s);
      Tensor q = qc.view({n, H, d});
      Tensor k = kc2.view({n, H, d});
      Tensor v = vc2.view({n, H, d});

      Tensor kc = k_cache_.slice(0, li, li + 1)
                      .view({H, cfg_.max_seq_len, d});
      Tensor vc = v_cache_.slice(0, li, li + 1)
                      .view({H, cfg_.max_seq_len, d});
      op::kv_append(kc, k, past_len, s);
      op::kv_append(vc, v, past_len, s);

      Tensor ao({n, H, d}, DType::kFloat16, ctx.device());
      Tensor sc = Tensor::wrap(scratch_.data_ptr(),
                               {static_cast<int64_t>(H) * n * kv_len},
                               DType::kFloat16, ctx.device(), "attn_scratch");
      op::attention(q, kc, vc, ao, H, H, past_len, true, sc, s);

      Tensor aof = ao.view({n, E});
      Tensor p1({n, E}, DType::kFloat16, ctx.device());
      op::linear_gemm(aof, for_gemm(L.proj), p1, GemmVersion::kV2Tiled, s);

      Tensor x2({n, E}, DType::kFloat16, ctx.device());
      op::add(x, p1, x2, s);
      x = std::move(x2);

      // --- MLP ---
      Tensor n2({n, E}, DType::kFloat16, ctx.device());
      op::layer_norm(x, L.ln2_w, &L.ln2_b, n2, cfg_.eps, s);

      Tensor h1({n, I}, DType::kFloat16, ctx.device());
      op::linear_gemm(n2, for_gemm(L.fc), h1, GemmVersion::kV2Tiled, s);
      Tensor act({n, I}, DType::kFloat16, ctx.device());
      op::gelu(h1, act, s);

      Tensor p2({n, E}, DType::kFloat16, ctx.device());
      op::linear_gemm(act, for_gemm(L.fcproj), p2, GemmVersion::kV2Tiled, s);

      Tensor x3({n, E}, DType::kFloat16, ctx.device());
      op::add(x, p2, x3, s);
      x = std::move(x3);
    }

    // 最终 LayerNorm，然后通过共享嵌入计算最后一个 token 的 logits。
    Tensor xf({n, E}, DType::kFloat16, ctx.device());
    op::layer_norm(x, lnf_w_, &lnf_b_, xf, cfg_.eps, s);

    Tensor last = xf.slice(0, n - 1, n).view({1, E});
    op::linear_gemv(last, {nullptr, &wte_, nullptr}, logits_, s);
    return logits_;
  }

 private:
  ModelConfig cfg_;
  Tensor wte_, wpe_;
  std::vector<GptLayer> layers_;
  Tensor lnf_w_, lnf_b_;
  Tensor k_cache_, v_cache_, scratch_, logits_;
};

}  // namespace

std::unique_ptr<LanguageModel> build_gpt2(
    std::unordered_map<std::string, Tensor> weights, const ModelConfig &cfg,
    Context &ctx) {
  return std::make_unique<GPT2Model>(std::move(weights), cfg, ctx);
}

}  // namespace cudallm
