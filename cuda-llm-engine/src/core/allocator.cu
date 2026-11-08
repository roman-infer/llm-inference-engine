#include "cuda_llm/allocator.h"

#include <algorithm>
#include <memory>

#include "cuda_llm/cuda_utils.h"
#include "cuda_llm/logger.h"

namespace cudallm {

// ---------------------------------------------------------------------------
// CachingAllocator
// ---------------------------------------------------------------------------
CachingAllocator::~CachingAllocator() { empty_cache(); }

void *CachingAllocator::allocate(size_t nbytes) {
  std::lock_guard<std::mutex> lock(mu_);
  if (nbytes == 0) return nullptr;

  // 按对齐要求向上取整。
  const size_t aligned =
      ((nbytes + kAlignment - 1) / kAlignment) * kAlignment;

  // 查找足够大的最小保留块。
  auto it = free_blocks_.lower_bound(aligned);
  if (it != free_blocks_.end()) {
    void *ptr = it->second;
    free_blocks_.erase(it);
    live_[ptr] = aligned;
    live_bytes_ += aligned;
    peak_ = std::max(peak_, live_bytes_);
    return ptr;
  }

  void *ptr = nullptr;
  CUDA_CHECK(cudaMalloc(&ptr, aligned));
  live_[ptr] = aligned;
  live_bytes_ += aligned;
  peak_ = std::max(peak_, live_bytes_);
  return ptr;
}

void CachingAllocator::release(void *ptr) {
  if (!ptr) return;
  std::lock_guard<std::mutex> lock(mu_);
  auto it = live_.find(ptr);
  if (it == live_.end()) {
    LOG_WARN("CachingAllocator::release on unknown pointer " << ptr);
    return;
  }
  const size_t size = it->second;
  live_.erase(it);
  live_bytes_ -= size;
  free_blocks_.emplace(size, ptr);
}

void CachingAllocator::empty_cache() {
  std::lock_guard<std::mutex> lock(mu_);
  for (auto &kv : free_blocks_) {
    // 程序退出时 CUDA 驱动可能已经在拆卸；此时 cudaFree 失败是无害的
    // （驱动会回收内存）。
    cudaError_t err = cudaFree(kv.second);
    if (err != cudaSuccess) {
      LOG_WARN("cudaFree during teardown: " << cudaGetErrorString(err));
    }
  }
  free_blocks_.clear();
}

// ---------------------------------------------------------------------------
// 进程默认分配器
// ---------------------------------------------------------------------------
Allocator *default_allocator() {
  static std::unique_ptr<Allocator> alloc =
      std::make_unique<CachingAllocator>();
  return alloc.get();
}

}  // namespace cudallm
