#include "cuda_llm/context.h"

#include "cuda_llm/cuda_utils.h"
#include "cuda_llm/logger.h"

namespace cudallm {

Context::Context(int device, size_t workspace_bytes)
    : device_(device) {
  CUDA_CHECK(cudaSetDevice(device_));
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device_));
  LOG_INFO("Context created on " << prop.name << " (sm_" << prop.major
                                 << prop.minor << ", "
                                 << prop.multiProcessorCount << " SMs)");

  workspace_ = default_allocator()->allocate(workspace_bytes);
  workspace_size_ = workspace_bytes;
}

Context::~Context() {
  if (workspace_) {
    default_allocator()->release(workspace_);
    workspace_ = nullptr;
  }
  if (stream_) {
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
  }
}

void *Context::workspace(size_t nbytes) {
  if (nbytes > workspace_size_) {
    // 以 64 MiB 为增量增长。
    size_t target = ((nbytes + (1ull << 26) - 1) / (1ull << 26)) *
                    (1ull << 26);
    default_allocator()->release(workspace_);
    workspace_ = default_allocator()->allocate(target);
    workspace_size_ = target;
    LOG_DEBUG("workspace grown to " << target / 1048576 << " MiB");
  }
  return workspace_;
}

}  // namespace cudallm
