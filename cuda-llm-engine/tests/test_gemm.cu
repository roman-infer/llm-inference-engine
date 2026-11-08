#include "cuda_llm/context.h"
#include "cuda_llm/gemm.h"
#include "cpu_ref.h"
#include "test_util.h"

using namespace cudallm;

namespace {

void compare_gemm_version(GemmVersion v, int M, int N, int K) {
  std::vector<float> A = cputest::quantize_inputs(
      cputest::random_float(M * K));
  std::vector<float> B = cputest::quantize_inputs(
      cputest::random_float(K * N));

  __half *Ad = cputest::upload_half(A);
  __half *Bd = cputest::upload_half(B);
  __half *Cd = nullptr;
  CUDA_CHECK(cudaMalloc(&Cd, M * N * sizeof(__half)));

  gemm_launch(v, Ad, Bd, Cd, M, N, K, 0);
  std::vector<float> got = cputest::download_half(Cd, M * N);
  std::vector<float> ref = cputest::gemm_ref(A, B, M, N, K);

  for (size_t i = 0; i < ref.size(); ++i) CHECK_CLOSE(got[i], ref[i]);

  cudaFree(Ad);
  cudaFree(Bd);
  cudaFree(Cd);
}

}  // namespace

void test_gemm_v0() { compare_gemm_version(GemmVersion::kV0Naive, 32, 64, 64); }
void test_gemm_v1() {
  compare_gemm_version(GemmVersion::kV1Vectorized, 64, 64, 128);
  compare_gemm_version(GemmVersion::kV1Vectorized, 40, 48, 32);
}
void test_gemm_v2() {
  compare_gemm_version(GemmVersion::kV2Tiled, 128, 128, 64);
  compare_gemm_version(GemmVersion::kV2Tiled, 130, 24, 40);
}

void test_gemm_v3() {
  const int M = 8, N = 96, K = 64;
  std::vector<float> A = cputest::quantize_inputs(
      cputest::random_float(M * K));
  std::vector<float> B = cputest::quantize_inputs(
      cputest::random_float(K * N));
  std::vector<float> bias = cputest::quantize_inputs(
      cputest::random_float(N, -0.2f, 0.2f));
  std::vector<float> gamma = cputest::quantize_inputs(
      cputest::random_float(N, 0.5f, 1.5f));

  __half *Ad = cputest::upload_half(A);
  __half *Bd = cputest::upload_half(B);
  __half *bd = cputest::upload_half(bias);
  __half *gd = cputest::upload_half(gamma);
  __half *Cd = nullptr;
  cudaMalloc(&Cd, M * N * sizeof(__half));

  gemm_bias_rmsnorm(Ad, Bd, bd, gd, Cd, M, N, K, 1e-5f, 0);
  std::vector<float> got = cputest::download_half(Cd, M * N);

  std::vector<float> cbias = cputest::gemm_ref(A, B, M, N, K);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) cbias[m * N + n] += bias[n];
  std::vector<float> ref =
      cputest::rmsnorm_ref(cbias, gamma, M, N, 1e-5f);

  for (size_t i = 0; i < ref.size(); ++i) CHECK_CLOSE(got[i], ref[i]);

  cudaFree(Ad); cudaFree(Bd); cudaFree(bd); cudaFree(gd); cudaFree(Cd);
}

void test_gemm_v4() {
  const int N = 128, K = 64;
  std::vector<float> x = cputest::quantize_inputs(
      cputest::random_float(K));
  std::vector<float> w = cputest::quantize_inputs(
      cputest::random_float(N * K));

  __half *xd = cputest::upload_half(x);
  __half *wd = cputest::upload_half(w);
  __half *yd = nullptr;
  cudaMalloc(&yd, N * sizeof(__half));

  gemv_launch(xd, wd, yd, K, N, 0);
  std::vector<float> got = cputest::download_half(yd, N);
  std::vector<float> ref = cputest::gemv_ref(x, w, N, K);

  for (int i = 0; i < N; ++i) CHECK_CLOSE(got[i], ref[i]);

  cudaFree(xd); cudaFree(wd); cudaFree(yd);
}

int main() {
  RUN_TEST(test_gemm_v0);
  RUN_TEST(test_gemm_v1);
  RUN_TEST(test_gemm_v2);
  RUN_TEST(test_gemm_v3);
  RUN_TEST(test_gemm_v4);
  return cputest::summarize("test_gemm");
}
