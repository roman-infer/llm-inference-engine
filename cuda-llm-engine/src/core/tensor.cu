#include "cuda_llm/tensor.h"

#include <numeric>
#include <stdexcept>

#include "cuda_llm/cuda_utils.h"

namespace cudallm {

// ---------------------------------------------------------------------------
// 数据类型辅助函数
// ---------------------------------------------------------------------------
size_t dtype_size(DType dtype) {
  switch (dtype) {
    case DType::kFloat32:
      return 4;
    case DType::kFloat16:
    case DType::kBFloat16:
      return 2;
    case DType::kInt32:
      return 4;
    case DType::kInt64:
      return 8;
    case DType::kInt8:
    case DType::kBool:
      return 1;
  }
  return 0;
}

DType dtype_from_safetensors(const std::string &name) {
  if (name == "F32") return DType::kFloat32;
  if (name == "F16") return DType::kFloat16;
  if (name == "BF16") return DType::kBFloat16;
  if (name == "I32") return DType::kInt32;
  if (name == "I64") return DType::kInt64;
  if (name == "I8") return DType::kInt8;
  if (name == "BOOL") return DType::kBool;
  throw std::runtime_error("Unsupported safetensors dtype: " + name);
}

// ---------------------------------------------------------------------------
// Tensor
// ---------------------------------------------------------------------------
Tensor::Tensor(std::vector<int64_t> shape, DType dtype, int device,
               std::string name)
    : shape_(std::move(shape)),
      dtype_(dtype),
      device_(device),
      name_(std::move(name)) {
  const size_t bytes = nbytes();
  if (bytes > 0) {
    data_ = default_allocator()->allocate(bytes);
    owns_ = true;
  }
}

Tensor Tensor::wrap(void *data, std::vector<int64_t> shape, DType dtype,
                    int device, std::string name) {
  Tensor t;
  t.data_ = data;
  t.shape_ = std::move(shape);
  t.dtype_ = dtype;
  t.device_ = device;
  t.owns_ = false;
  t.name_ = std::move(name);
  return t;
}

Tensor::~Tensor() { release(); }

Tensor::Tensor(Tensor &&other) noexcept
    : shape_(std::move(other.shape_)),
      dtype_(other.dtype_),
      data_(other.data_),
      owns_(other.owns_),
      device_(other.device_),
      name_(std::move(other.name_)) {
  other.data_ = nullptr;
  other.owns_ = false;
}

Tensor &Tensor::operator=(Tensor &&other) noexcept {
  if (this != &other) {
    release();
    shape_ = std::move(other.shape_);
    dtype_ = other.dtype_;
    data_ = other.data_;
    owns_ = other.owns_;
    device_ = other.device_;
    name_ = std::move(other.name_);
    other.data_ = nullptr;
    other.owns_ = false;
  }
  return *this;
}

int64_t Tensor::numel() const {
  if (shape_.empty()) return 0;
  int64_t n = 1;
  for (int64_t d : shape_) n *= d;
  return n;
}

size_t Tensor::nbytes() const {
  return static_cast<size_t>(numel()) * dtype_size(dtype_);
}

int64_t Tensor::size(int dim) const {
  if (dim < 0) dim += ndim();
  return shape_.at(static_cast<size_t>(dim));
}

int64_t Tensor::stride(int dim) const {
  if (dim < 0) dim += ndim();
  int64_t s = 1;
  for (int i = ndim() - 1; i > dim; --i) s *= shape_[i];
  return s;
}

Tensor Tensor::view(std::vector<int64_t> shape, std::string name) const {
  int64_t total = 1;
  for (int64_t d : shape) {
    if (d < 0) throw std::runtime_error("view: inferred dim not supported");
    total *= d;
  }
  if (total != numel()) {
    throw std::runtime_error("view: element count mismatch");
  }
  return wrap(data_, std::move(shape), dtype_, device_,
              name.empty() ? name_ : std::move(name));
}

Tensor Tensor::slice(int64_t dim, int64_t start, int64_t end,
                     std::string name) const {
  if (dim < 0) dim += ndim();
  const int64_t dim_size = size(dim);
  if (start < 0) start += dim_size;
  if (end < 0) end += dim_size;
  if (start > end || end > dim_size) {
    throw std::runtime_error("slice: invalid range");
  }

  std::vector<int64_t> new_shape = shape_;
  new_shape[dim] = end - start;

  const size_t offset_bytes =
      static_cast<size_t>(start * stride(dim)) * dtype_size(dtype_);
  void *new_data =
      data_ ? static_cast<char *>(data_) + offset_bytes : nullptr;

  return wrap(new_data, std::move(new_shape), dtype_, device_,
              name.empty() ? name_ : std::move(name));
}

void Tensor::zero_async(cudaStream_t stream) {
  if (data_ && nbytes() > 0) {
    CUDA_CHECK(cudaMemsetAsync(data_, 0, nbytes(), stream));
  }
}

void Tensor::release() {
  if (owns_ && data_) {
    default_allocator()->release(data_);
  }
  data_ = nullptr;
  owns_ = false;
}

}  // namespace cudallm
