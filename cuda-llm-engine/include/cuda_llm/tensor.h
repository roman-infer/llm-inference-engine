#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

#include "cuda_llm/allocator.h"

namespace cudallm {

enum class DType {
  kFloat32 = 0,
  kFloat16 = 1,
  kBFloat16 = 2,
  kInt32 = 3,
  kInt64 = 4,
  kInt8 = 5,
  kBool = 6,
};

size_t dtype_size(DType dtype);
// 解析 safetensors 数据类型字符串（"F16"、"F32"、"I32" 等）。
DType dtype_from_safetensors(const std::string &name);

// ---------------------------------------------------------------------------
// Tensor：形状 + 数据类型 + 设备指针。
//
// 拥有内存的 Tensor 通过进程默认分配器分配，析构时释放。
// 非拥有的 Tensor 通过 wrap()/view()/slice() 创建，仅引用外部内存
// （权重子块、KV cache 槽位）。
// ---------------------------------------------------------------------------
class Tensor {
 public:
  Tensor() = default;

  // 分配型构造函数。
  explicit Tensor(std::vector<int64_t> shape, DType dtype = DType::kFloat16,
                  int device = 0, std::string name = "");

  // 包装一个已存在的设备指针（非拥有）。
  static Tensor wrap(void *data, std::vector<int64_t> shape, DType dtype,
                     int device = 0, std::string name = "");

  ~Tensor();

  Tensor(const Tensor &) = delete;
  Tensor &operator=(const Tensor &) = delete;

  Tensor(Tensor &&other) noexcept;
  Tensor &operator=(Tensor &&other) noexcept;

  int64_t numel() const;
  size_t nbytes() const;
  int ndim() const { return static_cast<int>(shape_.size()); }
  int64_t size(int dim) const;
  int64_t stride(int dim) const;

  const std::vector<int64_t> &shape() const { return shape_; }
  DType dtype() const { return dtype_; }
  int device_id() const { return device_; }
  const std::string &name() const { return name_; }
  void set_name(std::string name) { name_ = std::move(name); }

  bool is_owning() const { return owns_; }

  void *data_ptr() const { return data_; }
  template <typename T>
  T *data_ptr() const {
    return reinterpret_cast<T *>(data_);
  }

  // 非拥有的 reshape；元素总数必须一致。
  Tensor view(std::vector<int64_t> shape, std::string name = "") const;

  // 沿指定维度做非拥有切片：[start, end)。
  Tensor slice(int64_t dim, int64_t start, int64_t end,
               std::string name = "") const;

  void zero_async(cudaStream_t stream = 0);

 private:
  void release();

  std::vector<int64_t> shape_{};
  DType dtype_ = DType::kFloat16;
  void *data_ = nullptr;
  bool owns_ = false;
  int device_ = 0;
  std::string name_;
};

}  // namespace cudallm
