#pragma once

#include <cuda_runtime.h>

#include <cstddef>

#include "cuda_llm/allocator.h"
#include "cuda_llm/cuda_utils.h"  // synchronize() 中使用了 CUDA_CHECK

namespace cudallm {

// 运行时上下文：持有 CUDA stream 和供多趟 kernel 使用的临时工作区
// （attention 分数、采样）。每次模型前向传播都接收一个 Context&。
class Context {
 public:
  explicit Context(int device = 0,
                   size_t workspace_bytes = 256ull * 1024 * 1024);
  ~Context();

  Context(const Context &) = delete;
  Context &operator=(const Context &) = delete;

  int device() const { return device_; }
  cudaStream_t stream() const { return stream_; }

  // 按需增长的临时缓冲区（设备端）。Kernel 将其作为工作区使用；
  // 不得在其他请求工作区的调用之间持有它。
  void *workspace(size_t nbytes);

  void synchronize() { CUDA_CHECK(cudaStreamSynchronize(stream_)); }

 private:
  int device_ = 0;
  cudaStream_t stream_{};
  void *workspace_ = nullptr;
  size_t workspace_size_ = 0;
};

}  // namespace cudallm
