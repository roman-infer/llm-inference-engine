#include "cuda_llm/gemm.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "cuda_llm/cuda_utils.h"

namespace cudallm {

const char *gemm_version_name(GemmVersion version) {
  switch (version) {
    case GemmVersion::kV0Naive:
      return "v0_naive";
    case GemmVersion::kV1Vectorized:
      return "v1_vectorized";
    case GemmVersion::kV2Tiled:
      return "v2_tiled";
    case GemmVersion::kV3EpilogueFusion:
      return "v3_epilogue_fusion";
    case GemmVersion::kV4GemvKvCache:
      return "v4_gemv_kvcache";
  }
  return "unknown";
}

void gemm_launch(GemmVersion version, const __half *A, const __half *B,
                 __half *C, int M, int N, int K, cudaStream_t stream) {
  switch (version) {
    case GemmVersion::kV0Naive:
      gemm_launch_v0(A, B, C, M, N, K, stream);
      break;
    case GemmVersion::kV1Vectorized:
      gemm_launch_v1(A, B, C, M, N, K, stream);
      break;
    case GemmVersion::kV2Tiled:
      gemm_launch_v2(A, B, C, M, N, K, stream);
      break;
    default:
      throw std::runtime_error(
          "gemm_launch: v3/v4 use gemm_bias_rmsnorm / gemv_launch");
  }
}

static float median_of(std::vector<float> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

float gemm_benchmark_ms(GemmVersion version, const __half *A, const __half *B,
                        __half *C, int M, int N, int K, cudaStream_t stream,
                        int iters) {
  // 预热（同时验证 kernel 正确性）。
  gemm_launch(version, A, B, C, M, N, K, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<float> times;
  times.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    EventTimer timer(stream);
    timer.start();
    gemm_launch(version, A, B, C, M, N, K, stream);
    times.push_back(timer.stop());
  }
  return median_of(times);
}

float gemv_benchmark_ms(const __half *x, const __half *w, __half *y, int K,
                        int N, cudaStream_t stream, int iters) {
  gemv_launch(x, w, y, K, N, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<float> times;
  times.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    EventTimer timer(stream);
    timer.start();
    gemv_launch(x, w, y, K, N, stream);
    times.push_back(timer.stop());
  }
  return median_of(times);
}

}  // namespace cudallm
