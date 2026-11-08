// GEMM / GEMV 性能测试。
//
// 形状按“实际负载类别”选取，而不是只挑对某个 kernel 有利的极端形状：
//   prefill   ：M 中等、N/K 为 4096 量级（线性层、qkv、MLP 的典型形状）
//   square    ：M=N=K，观察算力上限
//   narrow    ：N=8，刻意保留的退化形状——分块 kernel 在这里必输，
//               也是 decode 必须走 GEMV（v4）而不是 GEMM 的原因
//   decode    ：M=1，v4 的主场（带宽受限）
//
// 输出：毫秒、TFLOPS、相对 v0 的加速比；`--json <path>` 会额外落盘一份 JSON，
// 便于把数据贴进 README 或做回归对比。

#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_llm/context.h"
#include "cuda_llm/gemm.h"
#include "cuda_llm/ops.h"

using namespace cudallm;

namespace {

struct ShapeResult {
  std::string version;
  float ms = 0.0f;
  double tflops = 0.0;
  double speedup = 1.0;
};

struct ShapeBench {
  int M = 0, N = 0, K = 0;
  const char *kind = "";
  std::vector<ShapeResult> rows;
  bool has_gemv = false;
  float gemv_ms = 0.0f;
};

__half *device_half(const std::vector<float> &host) {
  std::vector<__half> h16(host.size());
  for (size_t i = 0; i < host.size(); ++i) h16[i] = __float2half(host[i]);
  __half *d = nullptr;
  CUDA_CHECK(cudaMalloc(&d, h16.size() * sizeof(__half)));
  CUDA_CHECK(cudaMemcpy(d, h16.data(), h16.size() * sizeof(__half),
                        cudaMemcpyHostToDevice));
  return d;
}

void bench_shape(Context &ctx, ShapeBench &sb, int iters) {
  const int M = sb.M, N = sb.N, K = sb.K;
  const size_t nA = static_cast<size_t>(M) * K;
  const size_t nB = static_cast<size_t>(K) * N;
  std::vector<float> hA(nA), hB(nB);
  std::mt19937 rng(1);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto &v : hA) v = dist(rng);
  for (auto &v : hB) v = dist(rng);

  __half *A = device_half(hA);
  __half *B = device_half(hB);
  __half *C = nullptr;
  CUDA_CHECK(cudaMalloc(&C, static_cast<size_t>(M) * N * sizeof(__half)));

  const GemmVersion versions[] = {GemmVersion::kV0Naive,
                                  GemmVersion::kV1Vectorized,
                                  GemmVersion::kV2Tiled};
  float base_ms = 0.0f;
  for (GemmVersion v : versions) {
    if (M == 1) break;  // decode 形状没有意义，交给 v4
    const float ms = gemm_benchmark_ms(v, A, B, C, M, N, K, ctx.stream(),
                                       iters);
    if (v == GemmVersion::kV0Naive) base_ms = ms;
    ShapeResult r;
    r.version = gemm_version_name(v);
    r.ms = ms;
    r.tflops = 2.0 * M * N * K / (ms * 1e9);
    r.speedup = base_ms / ms;
    sb.rows.push_back(r);
  }

  if (M == 1) {
    // GEMV 期望 w[N,K] 布局；为对比先转置 B。
    __half *w = nullptr;
    CUDA_CHECK(cudaMalloc(&w, nB * sizeof(__half)));
    Tensor Bt = Tensor::wrap(B, {K, N}, DType::kFloat16);
    Tensor Wt = Tensor::wrap(w, {N, K}, DType::kFloat16);
    op::transpose_2d(Bt, Wt, ctx.stream());

    __half *y = nullptr;
    CUDA_CHECK(cudaMalloc(&y, static_cast<size_t>(N) * sizeof(__half)));
    sb.gemv_ms = gemv_benchmark_ms(A, w, y, K, N, ctx.stream(), 50);
    sb.has_gemv = true;
    CUDA_CHECK(cudaFree(w));
    CUDA_CHECK(cudaFree(y));
  }

  CUDA_CHECK(cudaFree(A));
  CUDA_CHECK(cudaFree(B));
  CUDA_CHECK(cudaFree(C));
}

void print_table(const ShapeBench &sb) {
  std::printf("\n[%s] M=%d N=%d K=%d\n", sb.kind, sb.M, sb.N, sb.K);
  if (!sb.rows.empty()) {
    std::printf("  %-16s %10s %10s %10s\n", "version", "ms", "TFLOPS",
                "speedup");
    for (const auto &r : sb.rows) {
      std::printf("  %-16s %10.3f %10.2f %9.2fx\n", r.version.c_str(), r.ms,
                  r.tflops, r.speedup);
    }
  }
  if (sb.has_gemv) {
    // 带宽受限：bytes = N*K*2（权重） + N*2（输出），M=1 只读一次 x。
    const double gb = (2.0 * sb.N * sb.K + 2.0 * sb.N) / 1e9;
    const double gbps = gb / (sb.gemv_ms * 1e-3);
    std::printf("  %-16s %10.3f %10s %9.1f GB/s\n", "v4_gemv", sb.gemv_ms,
                "bw-bound", gbps);
  }
}

std::string json_escape(const std::string &s) { return s; }

}  // namespace

int main(int argc, char **argv) {
  std::string json_path;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--json") == 0 && i + 1 < argc) {
      json_path = argv[++i];
    }
  }

  Context ctx(0);
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

  std::printf("GPU: %s (sm_%d%d, %d SMs)\n", prop.name, prop.major,
              prop.minor, prop.multiProcessorCount);

  // 负载类别：prefill / square / narrow / decode
  struct Spec {
    int M, N, K;
    const char *kind;
    int iters;
  };
  const Spec specs[] = {
      {1024, 4096, 4096, "prefill", 10},
      {128, 4096, 4096, "prefill-small", 10},
      {2048, 2048, 2048, "square", 10},
      {128, 8, 8192, "narrow (degenerate)", 10},
      {1, 4096, 4096, "decode", 1},
      {1, 16384, 4096, "decode-large", 1},
  };

  std::vector<ShapeBench> all;
  for (const auto &sp : specs) {
    ShapeBench sb;
    sb.M = sp.M;
    sb.N = sp.N;
    sb.K = sp.K;
    sb.kind = sp.kind;
    bench_shape(ctx, sb, sp.iters);
    print_table(sb);
    all.push_back(std::move(sb));
  }

  if (!json_path.empty()) {
    FILE *f = std::fopen(json_path.c_str(), "w");
    if (!f) {
      std::fprintf(stderr, "cannot open %s\n", json_path.c_str());
      return 1;
    }
    std::fprintf(f, "{\n  \"device\": \"%s\",\n  \"capability\": \"%d.%d\",\n",
                 prop.name, prop.major, prop.minor);
    std::fprintf(f, "  \"cuda\": \"%d.%d\",\n", CUDART_VERSION / 1000,
                 (CUDART_VERSION % 1000) / 10);
    std::fprintf(f, "  \"shapes\": [\n");
    for (size_t i = 0; i < all.size(); ++i) {
      const auto &sb = all[i];
      std::fprintf(f,
                   "    {\"kind\": \"%s\", \"M\": %d, \"N\": %d, \"K\": %d, "
                   "\"rows\": [",
                   sb.kind, sb.M, sb.N, sb.K);
      for (size_t j = 0; j < sb.rows.size(); ++j) {
        const auto &r = sb.rows[j];
        std::fprintf(f,
                     "%s{\"version\": \"%s\", \"ms\": %.4f, \"tflops\": %.3f, "
                     "\"speedup\": %.3f}",
                     j ? ", " : "", json_escape(r.version).c_str(), r.ms,
                     r.tflops, r.speedup);
      }
      std::fprintf(f, "]");
      if (sb.has_gemv) {
        std::fprintf(f, ", \"gemv_ms\": %.4f", sb.gemv_ms);
      }
      std::fprintf(f, "}%s\n", i + 1 == all.size() ? "" : ",");
    }
    std::fprintf(f, "  ]\n}\n");
    std::fclose(f);
    std::printf("\nJSON written to %s\n", json_path.c_str());
  }
  return 0;
}

