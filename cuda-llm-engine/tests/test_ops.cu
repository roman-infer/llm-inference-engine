#include <vector>

#include "cuda_llm/context.h"
#include "cuda_llm/ops.h"
#include "cpu_ref.h"
#include "test_util.h"

using namespace cudallm;

namespace {

void fill_tensor(Tensor &t, const std::vector<float> &v,
                 cudaStream_t stream = 0) {
  std::vector<__half> h(v.size());
  for (size_t i = 0; i < v.size(); ++i) h[i] = __float2half(v[i]);
  CUDA_CHECK(cudaMemcpyAsync(t.data_ptr(), h.data(),
                             h.size() * sizeof(__half),
                             cudaMemcpyHostToDevice, stream));
}
std::vector<float> read_tensor(const Tensor &t, cudaStream_t stream = 0) {
  return cputest::download_half(t.data_ptr<__half>(), t.numel(), stream);
}

}  // namespace

void test_norms() {
  Context ctx;
  const int rows = 4, cols = 64;
  std::vector<float> x = cputest::quantize_inputs(
      cputest::random_float(rows * cols));
  std::vector<float> w = cputest::quantize_inputs(
      cputest::random_float(cols, 0.5f, 1.5f));
  std::vector<float> b = cputest::quantize_inputs(
      cputest::random_float(cols, -0.2f, 0.2f));

  Tensor xt({rows, cols}, DType::kFloat16, ctx.device());
  Tensor wt({cols}, DType::kFloat16, ctx.device());
  Tensor bt({cols}, DType::kFloat16, ctx.device());
  fill_tensor(xt, x, ctx.stream());
  fill_tensor(wt, w, ctx.stream());
  fill_tensor(bt, b, ctx.stream());

  Tensor yt({rows, cols}, DType::kFloat16, ctx.device());
  op::rms_norm(xt, wt, yt, 1e-5f, ctx.stream());
  auto got = read_tensor(yt, ctx.stream());
  auto ref = cputest::rmsnorm_ref(x, w, rows, cols, 1e-5f);
  for (size_t i = 0; i < ref.size(); ++i) CHECK_CLOSE(got[i], ref[i]);

  Tensor yl({rows, cols}, DType::kFloat16, ctx.device());
  op::layer_norm(xt, wt, &bt, yl, 1e-5f, ctx.stream());
  got = read_tensor(yl, ctx.stream());
  ref = cputest::layernorm_ref(x, w, &b, rows, cols, 1e-5f);
  for (size_t i = 0; i < ref.size(); ++i) CHECK_CLOSE(got[i], ref[i]);
}

void test_activations() {
  Context ctx;
  std::vector<float> x = cputest::quantize_inputs(
      cputest::random_float(256, -3.0f, 3.0f));
  Tensor xt({256}, DType::kFloat16, ctx.device());
  fill_tensor(xt, x, ctx.stream());

  Tensor yt({256}, DType::kFloat16, ctx.device());
  op::gelu(xt, yt, ctx.stream());
  auto got = read_tensor(yt, ctx.stream());
  for (size_t i = 0; i < x.size(); ++i)
    CHECK_CLOSE(got[i], cputest::gelu_ref(x[i]));

  op::silu(xt, yt, ctx.stream());
  got = read_tensor(yt, ctx.stream());
  for (size_t i = 0; i < x.size(); ++i)
    CHECK_CLOSE(got[i], cputest::silu_ref(x[i]));
}

void test_softmax() {
  Context ctx;
  const int rows = 3, cols = 128;
  std::vector<float> x = cputest::quantize_inputs(
      cputest::random_float(rows * cols));
  Tensor xt({rows, cols}, DType::kFloat16, ctx.device());
  fill_tensor(xt, x, ctx.stream());
  op::softmax(xt, rows, cols, ctx.stream());
  auto got = read_tensor(xt, ctx.stream());
  auto ref = cputest::softmax_ref(x, rows, cols);
  for (size_t i = 0; i < ref.size(); ++i) CHECK_CLOSE(got[i], ref[i]);
}

void test_rope() {
  Context ctx;
  const int seq = 4, heads = 2, d = 16;
  std::vector<float> q = cputest::quantize_inputs(
      cputest::random_float(seq * heads * d));
  std::vector<float> k = q;

  Tensor qt({seq, heads, d}, DType::kFloat16, ctx.device());
  Tensor kt({seq, heads, d}, DType::kFloat16, ctx.device());
  fill_tensor(qt, q, ctx.stream());
  fill_tensor(kt, k, ctx.stream());
  op::rope_inplace(qt, kt, 0, 10000.0f, ctx.stream());

  cputest::rope_ref(q, seq, heads, d, 0, 10000.0f);
  auto got = read_tensor(qt, ctx.stream());
  for (size_t i = 0; i < q.size(); ++i) CHECK_CLOSE(got[i], q[i]);
}

void test_embedding() {
  Context ctx;
  std::vector<float> table = cputest::quantize_inputs(
      cputest::random_float(10 * 8));
  std::vector<int> ids = {3, 0, 7};

  Tensor tt({10, 8}, DType::kFloat16, ctx.device());
  Tensor it({3}, DType::kInt32, ctx.device());
  fill_tensor(tt, table, ctx.stream());
  CUDA_CHECK(cudaMemcpyAsync(it.data_ptr(), ids.data(),
                             ids.size() * sizeof(int),
                             cudaMemcpyHostToDevice, ctx.stream()));

  Tensor out({3, 8}, DType::kFloat16, ctx.device());
  op::embedding(tt, it, out, ctx.stream());
  auto got = read_tensor(out, ctx.stream());
  for (int s = 0; s < 3; ++s)
    for (int d = 0; d < 8; ++d)
      CHECK_CLOSE(got[s * 8 + d], table[ids[s] * 8 + d]);
}

void test_transpose() {
  Context ctx;
  std::vector<float> x = cputest::quantize_inputs(
      cputest::random_float(4 * 8));
  Tensor xt({4, 8}, DType::kFloat16, ctx.device());
  Tensor yt({8, 4}, DType::kFloat16, ctx.device());
  fill_tensor(xt, x, ctx.stream());
  op::transpose_2d(xt, yt, ctx.stream());
  auto got = read_tensor(yt, ctx.stream());
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 8; ++c)
      CHECK_CLOSE(got[c * 4 + r], x[r * 8 + c]);
}

void test_greedy_sample() {
  Context ctx;
  std::vector<float> logits(16, 0.0f);
  logits[11] = 3.5f;
  logits[4] = 2.0f;

  Tensor lt({16}, DType::kFloat16, ctx.device());
  fill_tensor(lt, logits, ctx.stream());
  Tensor scratch({17}, DType::kFloat32, ctx.device());
  int id = op::sample(lt, 0.0f, 0, 1.0f, 0.5f, scratch, ctx.stream());
  CHECK_TRUE(id == 11);
}

namespace {

// CPU 参考：标准 scaled dot-product attention + 因果掩码。
// q[s][h][d]，k/v[kh][t][d]；past_len 之前的 kv 视为已缓存的历史。
std::vector<float> attention_ref(const std::vector<float> &q,
                                 const std::vector<float> &k,
                                 const std::vector<float> &v, int seq,
                                 int n_h, int n_kv, int kv_len, int d,
                                 int max_seq, int past_len, bool causal) {
  std::vector<float> out(static_cast<size_t>(seq) * n_h * d, 0.0f);
  const float scale = 1.0f / std::sqrt(static_cast<float>(d));
  std::vector<double> sc(kv_len);
  for (int s = 0; s < seq; ++s) {
    for (int h = 0; h < n_h; ++h) {
      const int kh = h * n_kv / n_h;
      double mx = -1e30;
      for (int t = 0; t < kv_len; ++t) {
        const bool valid = !causal || t <= past_len + s;
        double acc = 0.0;
        if (valid) {
          for (int i = 0; i < d; ++i)
            acc += static_cast<double>(q[(s * n_h + h) * d + i]) *
                   k[(kh * max_seq + t) * d + i];
          acc *= scale;
        } else {
          acc = -1e30;
        }
        sc[t] = acc;
        if (acc > mx) mx = acc;
      }
      double sum = 0.0;
      for (int t = 0; t < kv_len; ++t) {
        sc[t] = std::exp(sc[t] - mx);
        sum += sc[t];
      }
      for (int i = 0; i < d; ++i) {
        double acc = 0.0;
        for (int t = 0; t < kv_len; ++t) {
          acc += sc[t] * v[(kh * max_seq + t) * d + i];
        }
        out[(s * n_h + h) * d + i] = static_cast<float>(acc / sum);
      }
    }
  }
  return out;
}

void run_attention_case(int seq, int n_h, int n_kv, int d, int max_seq,
                        int past_len, bool causal) {
  Context ctx;
  const int kv_len = past_len + seq;
  const size_t nq = static_cast<size_t>(seq) * n_h * d;
  const size_t nkv = static_cast<size_t>(n_kv) * max_seq * d;

  std::vector<float> q = cputest::quantize_inputs(cputest::random_float(nq));
  std::vector<float> k = cputest::quantize_inputs(cputest::random_float(nkv));
  std::vector<float> v =
      cputest::quantize_inputs(cputest::random_float(nkv, 0.0f, 2.0f));

  Tensor qt({seq, n_h, d}, DType::kFloat16, ctx.device());
  Tensor kt({n_kv, max_seq, d}, DType::kFloat16, ctx.device());
  Tensor vt({n_kv, max_seq, d}, DType::kFloat16, ctx.device());
  fill_tensor(qt, q, ctx.stream());
  fill_tensor(kt, k, ctx.stream());
  fill_tensor(vt, v, ctx.stream());

  Tensor scratch({static_cast<int64_t>(n_h) * seq * kv_len},
                 DType::kFloat16, ctx.device());
  Tensor out_ref({seq, n_h, d}, DType::kFloat16, ctx.device());
  Tensor out_onl({seq, n_h, d}, DType::kFloat16, ctx.device());

  op::attention(qt, kt, vt, out_ref, n_h, n_kv, past_len, causal, scratch,
                ctx.stream());
  op::attention_online(qt, kt, vt, out_onl, n_h, n_kv, past_len, causal,
                       ctx.stream());
  auto a = read_tensor(out_ref, ctx.stream());
  auto b = read_tensor(out_onl, ctx.stream());
  auto r = attention_ref(q, k, v, seq, n_h, n_kv, kv_len, d, max_seq,
                         past_len, causal);

  for (size_t i = 0; i < r.size(); ++i) {
    CHECK_CLOSE(a[i], r[i]);
    CHECK_CLOSE(b[i], r[i]);
    CHECK_CLOSE(b[i], a[i]);
  }
}

}  // namespace

// 三趟参考实现 与 在线 softmax 两条路径都要对得上 CPU 参考；
// 覆盖 MHA / GQA、纯 prefill 与带 past_len 的 decode 两种形态。
void test_attention() {
  run_attention_case(4, 2, 2, 16, 8, 0, true);   // MHA, prefill
  run_attention_case(3, 4, 2, 32, 12, 5, true);  // GQA, 带历史
  run_attention_case(1, 8, 8, 64, 16, 7, true);  // decode 单 token
  run_attention_case(2, 2, 1, 32, 8, 0, false);  // MQA, 无掩码
}

int main() {
  RUN_TEST(test_norms);
  RUN_TEST(test_activations);
  RUN_TEST(test_softmax);
  RUN_TEST(test_rope);
  RUN_TEST(test_embedding);
  RUN_TEST(test_transpose);
  RUN_TEST(test_greedy_sample);
  RUN_TEST(test_attention);
  return cputest::summarize("test_ops");
}
