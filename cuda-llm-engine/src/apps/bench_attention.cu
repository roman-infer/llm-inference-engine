// Attention 两条路径对比：三趟参考实现 vs 在线 softmax。
//
// 关注的不是“谁更快”这一句话，而是三个量：
//   ms       —— 延迟
//   scratch  —— 三趟实现必须物化的 [n_heads*seq, kv_len] 分数矩阵（字节）
//   有效显存 —— 在线版本的中间量只有每行 (m, l, o)，O(seq*d)
//
// 典型结论（RTX 4060 Laptop, seq=128, H=12, d=64）：
//   短 kv（<=512）  三趟更快：并行度铺在 kv 维上
//   长 kv（>=2048） 在线更快：三趟要写+读 12MB 以上的分数矩阵
// 这也解释了 FlashAttention 为什么在高并发、长上下文场景才是刚需。
//
// `--json <path>` 落盘，便于贴进 README。

#include <algorithm>
#include <cstdio>
#include <functional>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_llm/context.h"
#include "cuda_llm/ops.h"

using namespace cudallm;

namespace {

float bench_ms(std::function<void()> fn, int iters, cudaStream_t stream) {
  // 预热
  fn();
  CUDA_CHECK(cudaStreamSynchronize(stream));
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  std::vector<float> ts;
  ts.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    CUDA_CHECK(cudaEventRecord(a, stream));
    fn();
    CUDA_CHECK(cudaEventRecord(b, stream));
    CUDA_CHECK(cudaEventSynchronize(b));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    ts.push_back(ms);
  }
  CUDA_CHECK(cudaEventDestroy(a));
  CUDA_CHECK(cudaEventDestroy(b));
  std::sort(ts.begin(), ts.end());
  return ts[ts.size() / 2];
}

struct Row {
  int seq, heads, kv, d;
  float ref_ms, online_ms;
  double scratch_mb;
};

}  // namespace

int main(int argc, char **argv) {
  std::string json_path;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--json") == 0 && i + 1 < argc) {
      json_path = argv[++i];
    }
  }

  Context ctx(0);
  cudaStream_t s = ctx.stream();
  const int H = 12, KH = 12, D = 64;

  const int seqs[] = {1, 128};
  const int kvs[] = {128, 512, 1024, 2048, 4096};

  std::printf("GPU attention: H=%d KVH=%d d=%d\n", H, KH, D);
  std::printf("  %6s %8s %10s %10s %12s\n", "seq", "kv", "3pass ms",
              "online ms", "scratch MB");

  std::vector<Row> rows;
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (int seq : seqs) {
    for (int kv : kvs) {
      if (kv < seq) continue;
      const int past = kv - seq;
      const int max_seq = kv;
      const size_t nq = static_cast<size_t>(seq) * H * D;
      const size_t nkv = static_cast<size_t>(KH) * max_seq * D;

      std::vector<__half> hq(nq), hk(nkv), hv(nkv);
      for (auto &v : hq) v = __float2half(dist(rng));
      for (auto &v : hk) v = __float2half(dist(rng));
      for (auto &v : hv) v = __float2half(dist(rng));

      __half *dq = nullptr, *dk = nullptr, *dv = nullptr, *dout = nullptr;
      CUDA_CHECK(cudaMalloc(&dq, nq * sizeof(__half)));
      CUDA_CHECK(cudaMalloc(&dk, nkv * sizeof(__half)));
      CUDA_CHECK(cudaMalloc(&dv, nkv * sizeof(__half)));
      CUDA_CHECK(cudaMalloc(&dout, nq * sizeof(__half)));
      CUDA_CHECK(cudaMemcpyAsync(dq, hq.data(), nq * sizeof(__half),
                                 cudaMemcpyHostToDevice, s));
      CUDA_CHECK(cudaMemcpyAsync(dk, hk.data(), nkv * sizeof(__half),
                                 cudaMemcpyHostToDevice, s));
      CUDA_CHECK(cudaMemcpyAsync(dv, hv.data(), nkv * sizeof(__half),
                                 cudaMemcpyHostToDevice, s));
      CUDA_CHECK(cudaStreamSynchronize(s));

      Tensor q = Tensor::wrap(dq, {seq, H, D}, DType::kFloat16, 0);
      Tensor kc = Tensor::wrap(dk, {KH, max_seq, D}, DType::kFloat16, 0);
      Tensor vc = Tensor::wrap(dv, {KH, max_seq, D}, DType::kFloat16, 0);
      Tensor out = Tensor::wrap(dout, {seq, H, D}, DType::kFloat16, 0);

      const size_t scratch_elems =
          static_cast<size_t>(H) * seq * kv;
      __half *dscratch = nullptr;
      CUDA_CHECK(cudaMalloc(&dscratch, scratch_elems * sizeof(__half)));
      Tensor scratch = Tensor::wrap(dscratch, {static_cast<int64_t>(H) * seq, kv},
                                    DType::kFloat16, 0);

      const float ref_ms = bench_ms(
          [&] { op::attention(q, kc, vc, out, H, KH, past, true, scratch, s); },
          20, s);
      const float onl_ms = bench_ms(
          [&] { op::attention_online(q, kc, vc, out, H, KH, past, true, s); },
          20, s);

      Row r{seq, H, kv, D, ref_ms, onl_ms,
            static_cast<double>(scratch_elems * sizeof(__half)) / 1e6};
      rows.push_back(r);
      std::printf("  %6d %8d %10.3f %10.3f %12.2f\n", seq, kv, ref_ms, onl_ms,
                  r.scratch_mb);

      CUDA_CHECK(cudaFree(dscratch));
      CUDA_CHECK(cudaFree(dq));
      CUDA_CHECK(cudaFree(dk));
      CUDA_CHECK(cudaFree(dv));
      CUDA_CHECK(cudaFree(dout));
    }
  }

  if (!json_path.empty()) {
    FILE *f = std::fopen(json_path.c_str(), "w");
    if (!f) return 1;
    std::fprintf(f, "{\n  \"heads\": %d,\n  \"kv_heads\": %d,\n  \"head_dim\": %d,\n  \"rows\": [\n", H, KH, D);
    for (size_t i = 0; i < rows.size(); ++i) {
      const auto &r = rows[i];
      std::fprintf(f,
                   "    {\"seq\": %d, \"kv\": %d, \"ms_3pass\": %.4f, "
                   "\"ms_online\": %.4f, \"scratch_mb\": %.3f}%s\n",
                   r.seq, r.kv, r.ref_ms, r.online_ms, r.scratch_mb,
                   i + 1 == rows.size() ? "" : ",");
    }
    std::fprintf(f, "  ]\n}\n");
    std::fclose(f);
    std::printf("\nJSON written to %s\n", json_path.c_str());
  }
  return 0;
}

