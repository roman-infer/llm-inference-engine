#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <map>
#include <mutex>
#include <unordered_map>

namespace cudallm {

// 抽象的设备内存分配器。Tensor 和 Context 都通过此接口分配内存，
// 这样可以在不修改 kernel 代码的情况下切换分配策略
// （直接 cudaMalloc vs 缓存池）。
class Allocator {
 public:
  virtual ~Allocator() = default;

  virtual void *allocate(size_t nbytes) = 0;
  virtual void release(void *ptr) = 0;

  virtual size_t allocated_bytes() const = 0;
  virtual size_t peak_bytes() const = 0;
};

// 池化分配器。释放的内存块按大小排序保存在空闲链表中，下次直接复用，
// 从而消除 decode 循环中 cudaMalloc 的同步开销。
// 每个块按 kAlignment 字节对齐向上取整。
class CachingAllocator : public Allocator {
 public:
  ~CachingAllocator() override;

  void *allocate(size_t nbytes) override;
  void release(void *ptr) override;

  size_t allocated_bytes() const override { return live_bytes_; }
  size_t peak_bytes() const override { return peak_; }

  // 将所有保留（未使用）的块释放回驱动程序。
  void empty_cache();

 private:
  static constexpr size_t kAlignment = 256;

  std::multimap<size_t, void *> free_blocks_;
  std::unordered_map<void *, size_t> live_;
  size_t live_bytes_ = 0;
  size_t peak_ = 0;
  std::mutex mu_;
};

// 进程级默认分配器（一个 CachingAllocator）。
Allocator *default_allocator();

}  // namespace cudallm
