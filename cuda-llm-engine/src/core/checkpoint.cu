#include "cuda_llm/checkpoint.h"

#include <fstream>
#include <utility>

#include "cuda_llm/cuda_utils.h"
#include "cuda_llm/json_min.h"
#include "cuda_llm/logger.h"
#include "cuda_llm/ops.h"

namespace cudallm {

Checkpoint::Checkpoint(std::string path) : path_(std::move(path)) {
  std::ifstream file(path_, std::ios::binary | std::ios::ate);
  if (!file) throw std::runtime_error("Cannot open checkpoint: " + path_);

  const std::streamsize size = file.tellg();
  if (size < 8) throw std::runtime_error("safetensors: file too small");
  file.seekg(0, std::ios::beg);

  file_bytes_.resize(static_cast<size_t>(size));
  if (!file.read(file_bytes_.data(), size)) {
    throw std::runtime_error("safetensors: read failed");
  }

  parse_header();
  LOG_INFO("Loaded safetensors " << path_ << " (" << size / 1048576 << " MiB, "
                                 << keys_.size() << " tensors)");
}

void Checkpoint::parse_header() {
  uint64_t header_len = 0;
  for (int i = 0; i < 8; ++i) {
    header_len |= static_cast<uint64_t>(
                      static_cast<unsigned char>(file_bytes_[i]))
                  << (8 * i);
  }
  if (8 + header_len > file_bytes_.size()) {
    throw std::runtime_error("safetensors: bad header length");
  }

  const std::string header(file_bytes_.data() + 8,
                           static_cast<size_t>(header_len));
  data_offset_ = 8 + static_cast<size_t>(header_len);

  JsonValue root = JsonParser::parse(header);
  if (root.type != JsonValue::kObject) {
    throw std::runtime_error("safetensors: header is not an object");
  }

  for (const auto &kv : root.obj) {
    const std::string &name = kv.first;
    const JsonValue &v = kv.second;
    if (name == "__metadata__") continue;

    TensorMeta meta;
    meta.dtype = dtype_from_safetensors(v.at("dtype").as_string());
    for (const JsonValue &d : v.at("shape").as_array()) {
      meta.shape.push_back(d.as_i64());
    }
    const auto &offsets = v.at("data_offsets").as_array();
    meta.offset_begin = static_cast<size_t>(offsets.at(0).as_i64());
    meta.offset_end = static_cast<size_t>(offsets.at(1).as_i64());

    keys_.push_back(name);
    metas_.emplace(name, std::move(meta));
  }
}

Tensor Checkpoint::load_tensor(const std::string &key, Context &ctx,
                               DType target_dtype) {
  const TensorMeta &m = metas_.at(key);
  const size_t raw_bytes = m.offset_end - m.offset_begin;
  const char *host_src =
      file_bytes_.data() + data_offset_ + m.offset_begin;
  cudaStream_t stream = ctx.stream();

  if (m.dtype == target_dtype) {
    Tensor out(m.shape, target_dtype, ctx.device(), key);
    CUDA_CHECK(cudaMemcpyAsync(out.data_ptr(), host_src, raw_bytes,
                               cudaMemcpyHostToDevice, stream));
    return out;
  }

  // F32 -> F16：先拷贝到设备，再用转换 kernel 处理。
  if (m.dtype == DType::kFloat32 && target_dtype == DType::kFloat16) {
    Tensor tmp(m.shape, DType::kFloat32, ctx.device(), key + "_f32");
    CUDA_CHECK(cudaMemcpyAsync(tmp.data_ptr(), host_src, raw_bytes,
                               cudaMemcpyHostToDevice, stream));
    Tensor out(m.shape, DType::kFloat16, ctx.device(), key);
    op::cast_f32_to_f16(tmp.data_ptr<float>(), out.data_ptr<__half>(),
                        out.numel(), stream);
    return out;
  }

  if (m.dtype == DType::kFloat16 && target_dtype == DType::kFloat32) {
    Tensor tmp(m.shape, DType::kFloat16, ctx.device(), key + "_f16");
    CUDA_CHECK(cudaMemcpyAsync(tmp.data_ptr(), host_src, raw_bytes,
                               cudaMemcpyHostToDevice, stream));
    Tensor out(m.shape, DType::kFloat32, ctx.device(), key);
    op::cast_f16_to_f32(tmp.data_ptr<__half>(), out.data_ptr<float>(),
                        out.numel(), stream);
    return out;
  }

  throw std::runtime_error("Unsupported dtype conversion for " + key);
}

std::unordered_map<std::string, Tensor> Checkpoint::load_all(
    Context &ctx, DType target_dtype) {
  std::unordered_map<std::string, Tensor> weights;
  for (const std::string &key : keys_) {
    weights.emplace(key, load_tensor(key, ctx, target_dtype));
  }
  return weights;
}

}  // namespace cudallm
