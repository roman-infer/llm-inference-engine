#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "cuda_llm/cuda_utils.h"

namespace cputest {

// 所有 kernel 使用的比较容差：atol=1e-2 / rtol=1e-2。
inline bool close(float a, float b, float atol = 1e-2f,
                  float rtol = 1e-2f) {
  return std::fabs(a - b) <= atol + rtol * std::fabs(b);
}

inline std::vector<float> random_float(size_t n, float lo = -1.0f,
                                       float hi = 1.0f,
                                       unsigned seed = 0x1234) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> v(n);
  for (auto &x : v) x = dist(rng);
  return v;
}

// 经过 half 量化，使 CPU 参考实现看到与 GPU 相同的输入。
inline std::vector<float> quantize_inputs(const std::vector<float> &v) {
  std::vector<float> q(v.size());
  for (size_t i = 0; i < v.size(); ++i) q[i] = __half2float(__float2half(v[i]));
  return q;
}

inline __half *upload_half(const std::vector<float> &v,
                           cudaStream_t stream = 0) {
  std::vector<__half> h(v.size());
  for (size_t i = 0; i < v.size(); ++i) h[i] = __float2half(v[i]);
  __half *d = nullptr;
  CUDA_CHECK(cudaMalloc(&d, h.size() * sizeof(__half)));
  CUDA_CHECK(cudaMemcpyAsync(d, h.data(), h.size() * sizeof(__half),
                             cudaMemcpyHostToDevice, stream));
  return d;
}

inline void upload_in_place(void *dst, const void *src, size_t bytes,
                            cudaStream_t stream = 0) {
  CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice,
                             stream));
}

inline std::vector<float> download_half(const __half *d, size_t n,
                                        cudaStream_t stream = 0) {
  std::vector<__half> h(n);
  CUDA_CHECK(cudaMemcpyAsync(h.data(), d, n * sizeof(__half),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::vector<float> out(n);
  for (size_t i = 0; i < n; ++i) out[i] = __half2float(h[i]);
  return out;
}

// 简单的测试注册器 / 断言宏。
struct TestState {
  int checks = 0;
  int failures = 0;
  std::string current;
};
inline TestState &state() {
  static TestState s;
  return s;
}

#define RUN_TEST(fn)                                                          \
  do {                                                                        \
    cputest::state().current = #fn;                                           \
    std::printf("[ RUN  ] %s\n", #fn);                                        \
    fn();                                                                     \
  } while (0)

#define CHECK_TRUE(cond)                                                      \
  do {                                                                        \
    cputest::state().checks++;                                                 \
    if (!(cond)) {                                                            \
      cputest::state().failures++;                                             \
      std::printf("  FAIL %s:%d %s\n", __FILE__, __LINE__, #cond);            \
    }                                                                         \
  } while (0)

#define CHECK_CLOSE(a, b)                                                     \
  do {                                                                        \
    cputest::state().checks++;                                                 \
    float __a = static_cast<float>(a);                                        \
    float __b = static_cast<float>(b);                                        \
    if (!cputest::close(__a, __b)) {                                          \
      cputest::state().failures++;                                             \
      std::printf("  FAIL %s:%d |%g - %g|=%g\n", __FILE__, __LINE__, __a,     \
                  __b, std::fabs(__a - __b));                                 \
    }                                                                         \
  } while (0)

inline int summarize(const char *suite) {
  auto &s = state();
  std::printf("\n%s: %d checks, %d failures\n", suite, s.checks,
              s.failures);
  return s.failures == 0 ? 0 : 1;
}

}  // namespace cputest
