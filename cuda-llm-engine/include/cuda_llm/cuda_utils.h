#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// 错误检查宏
// ---------------------------------------------------------------------------
#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t __err = (expr);                                                \
    if (__err != cudaSuccess) {                                                \
      std::fprintf(stderr, "[CUDA ERROR] %s:%d\n  %s\n  %s\n", __FILE__,       \
                   __LINE__, #expr, cudaGetErrorString(__err));                \
      throw std::runtime_error(std::string("CUDA error: ") +                   \
                               cudaGetErrorString(__err));                      \
    }                                                                          \
  } while (0)

// 启动 kernel 并立即检查启动/配置错误。
#define CUDA_LAUNCH(kernel, grid, block, shm_bytes, stream, ...)               \
  do {                                                                         \
    kernel<<<grid, block, shm_bytes, stream>>>(__VA_ARGS__);                   \
    CUDA_CHECK(cudaGetLastError());                                            \
  } while (0)

namespace cudallm {

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// ---------------------------------------------------------------------------
// 基于 CUDA Event 的计时器。所有 GEMM 加速比测试都使用它，
// 以保证 v0..v4 的测量方法一致。
// ---------------------------------------------------------------------------
class EventTimer {
 public:
  explicit EventTimer(cudaStream_t stream = 0) : stream_(stream) {
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&stop_));
  }

  ~EventTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  EventTimer(const EventTimer &) = delete;
  EventTimer &operator=(const EventTimer &) = delete;

  void start() { CUDA_CHECK(cudaEventRecord(start_, stream_)); }

  // 返回 start() 到 stop() 之间的毫秒数。会同步。
  float stop() {
    CUDA_CHECK(cudaEventRecord(stop_, stream_));
    CUDA_CHECK(cudaEventSynchronize(stop_));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
    return ms;
  }

 private:
  cudaEvent_t start_{};
  cudaEvent_t stop_{};
  cudaStream_t stream_{};
};

}  // namespace cudallm
