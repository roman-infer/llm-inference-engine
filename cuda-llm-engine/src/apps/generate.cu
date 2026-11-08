// 文本生成演示。
//
// 示例：
//   generate --arch gpt2 --ckpt models/gpt2/model.safetensors \
//            --tok models/gpt2 --prompt "The meaning of life is" \
//            --max_new 64 --temp 0.8 --top_k 50 --top_p 0.95

#include <chrono>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "cuda_llm/checkpoint.h"
#include "cuda_llm/context.h"
#include "cuda_llm/json_min.h"
#include "cuda_llm/model.h"
#include "cuda_llm/ops.h"
#include "cuda_llm/tokenizer.h"

using namespace cudallm;

static std::string arg_value(int argc, char **argv, const std::string &flag,
                             const std::string &def) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (flag == argv[i]) return argv[i + 1];
  }
  return def;
}

// 从模型目录的 config.json 读取配置（覆盖默认值），使 tiny / 自定义模型可用。
static bool load_config_from_json(const std::string &path, ModelConfig &cfg) {
  std::ifstream f(path);
  if (!f.is_open()) return false;
  std::stringstream ss;
  ss << f.rdbuf();
  try {
    JsonValue j = JsonParser::parse(ss.str());
    if (j.contains("n_layer")) cfg.n_layer = j.at("n_layer").as_int();
    if (j.contains("n_head")) cfg.n_head = j.at("n_head").as_int();
    if (j.contains("n_kv_head")) cfg.n_kv_head = j.at("n_kv_head").as_int();
    if (j.contains("n_embd")) cfg.n_embd = j.at("n_embd").as_int();
    if (j.contains("n_inter")) cfg.n_inter = j.at("n_inter").as_int();
    if (j.contains("max_seq_len"))
      cfg.max_seq_len = j.at("max_seq_len").as_int();
    if (j.contains("vocab_size")) cfg.vocab_size = j.at("vocab_size").as_int();
    if (j.contains("eps"))
      cfg.eps = static_cast<float>(j.at("eps").as_number());
    if (j.contains("theta"))
      cfg.theta = static_cast<float>(j.at("theta").as_number());
    return true;
  } catch (...) {
    return false;
  }
}

static Tensor ids_to_device(const std::vector<int> &ids, Context &ctx) {
  Tensor t({static_cast<int64_t>(ids.size())}, DType::kInt32,
           ctx.device(), "ids");
  CUDA_CHECK(cudaMemcpyAsync(t.data_ptr(), ids.data(),
                             ids.size() * sizeof(int),
                             cudaMemcpyHostToDevice, ctx.stream()));
  return t;
}

int main(int argc, char **argv) {
  const std::string arch = arg_value(argc, argv, "--arch", "gpt2");
  const std::string ckpt_path =
      arg_value(argc, argv, "--ckpt", "models/gpt2/model.safetensors");
  const std::string tok_dir =
      arg_value(argc, argv, "--tok", "models/gpt2");
  const std::string prompt =
      arg_value(argc, argv, "--prompt", "Once upon a time");
  const int max_new = std::stoi(arg_value(argc, argv, "--max_new", "64"));
  const float temp = std::stof(arg_value(argc, argv, "--temp", "0.8"));
  const int top_k = std::stoi(arg_value(argc, argv, "--top_k", "50"));
  const float top_p = std::stof(arg_value(argc, argv, "--top_p", "0.95"));

  Context ctx(0);

  Checkpoint ckpt(ckpt_path);
  auto weights = ckpt.load_all(ctx, DType::kFloat16);

  ModelConfig cfg;
  if (arch == "gpt2") {
    cfg = {12, 12, 12, 768, 3072, 1024, 50257, 1e-5f, 10000.0f,
           DType::kFloat16};
  } else {
    cfg = {6, 12, 4, 768, 2048, 1024, 32000, 1e-5f, 10000.0f,
           DType::kFloat16};
  }
  // 若模型目录下存在 config.json（如 tiny 模型），用其覆盖默认配置。
  const std::string cfg_path = tok_dir + "/config.json";
  if (load_config_from_json(cfg_path, cfg)) {
    std::printf("Loaded config from %s: L=%d H=%d KH=%d E=%d I=%d S=%d V=%d\n",
                cfg_path.c_str(), cfg.n_layer, cfg.n_head, cfg.n_kv_head,
                cfg.n_embd, cfg.n_inter, cfg.max_seq_len, cfg.vocab_size);
  }

  std::unique_ptr<LanguageModel> model =
      (arch == "gpt2")
          ? build_gpt2(std::move(weights), cfg, ctx)
          : build_llama(std::move(weights), cfg, ctx);

  BPETokenizer tokenizer = BPETokenizer::from_dir(tok_dir);

  std::vector<int> ids = tokenizer.encode(prompt);
  const size_t prompt_len = ids.size();
  std::printf("Prompt: %s (%zu tokens)\n", prompt.c_str(), prompt_len);
  std::printf("----------\n%s", prompt.c_str());
  std::fflush(stdout);

  std::mt19937 rng(0xC0FFEE);
  std::uniform_real_distribution<float> uniform(0.0f, 1.0f);

  // 采样临时工作区（float 工作区：vocab*4 + 4）。
  Tensor sample_scratch({cfg.vocab_size + 1}, DType::kFloat32,
                        ctx.device(), "sample_scratch");

  using clock = std::chrono::high_resolution_clock;
  double prefill_ms = 0.0, decode_ms = 0.0;

  // Prefill 阶段（一次性处理整个 prompt）。
  {
    auto t0 = clock::now();
    Tensor ids_d = ids_to_device(ids, ctx);
    const Tensor &logits =
        model->forward(ids_d, static_cast<int>(ids.size()), 0, ctx);
    const int next = op::sample(logits, temp, top_k, top_p, uniform(rng),
                                sample_scratch, ctx.stream());
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
    auto t1 = clock::now();
    prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    ids.push_back(next);
    std::printf("%s", tokenizer.decode({next}).c_str());
    std::fflush(stdout);
  }

  // Decode 循环（逐 token 生成）。
  {
    auto t0 = clock::now();
    for (int step = 0; step < max_new - 1; ++step) {
      const int past = static_cast<int>(ids.size()) - 1;
      std::vector<int> one{ids.back()};
      Tensor ids_d = ids_to_device(one, ctx);

      const Tensor &logits = model->forward(ids_d, 1, past, ctx);
      const int next = op::sample(logits, temp, top_k, top_p, uniform(rng),
                                  sample_scratch, ctx.stream());
      if (next == tokenizer.eos_id()) break;

      ids.push_back(next);
      std::printf("%s", tokenizer.decode({next}).c_str());
      std::fflush(stdout);
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
    auto t1 = clock::now();
    decode_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  }

  const size_t generated = ids.size() - prompt_len;
  const size_t decode_tokens = (generated > 0) ? generated - 1 : 0;
  std::printf("\n----------\n");
  std::printf("Generated %zu tokens (prompt %zu + new %zu)\n", ids.size(),
              prompt_len, generated);
  std::printf("Prefill:  %.2f ms (%zu prompt tokens)\n", prefill_ms,
              prompt_len);
  if (decode_tokens > 0) {
    std::printf("Decode:   %.2f ms total, %.2f ms/token, %.2f tokens/s\n",
                decode_ms, decode_ms / decode_tokens,
                decode_tokens / (decode_ms / 1000.0));
  }
  return 0;
}
