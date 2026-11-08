#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "cuda_llm/context.h"
#include "cuda_llm/tensor.h"

namespace cudallm {

struct TensorMeta {
  DType dtype = DType::kFloat16;
  std::vector<int64_t> shape;
  // 相对于数据段起始位置的字节偏移。
  size_t offset_begin = 0;
  size_t offset_end = 0;
};

// safetensors 格式加载器：
//   [8 字节小端序头部长度 N][N 字节 JSON 头部][原始数据]
// 整个文件一次性读入主机内存；随后按需将张量拷贝到设备
// （可选进行 F32<->F16 转换）。
class Checkpoint {
 public:
  explicit Checkpoint(std::string path);

  const std::vector<std::string> &keys() const { return keys_; }

  // 按名称加载一个张量。target_dtype 强制进行类型转换（F32<->F16）。
  Tensor load_tensor(const std::string &key, Context &ctx,
                     DType target_dtype = DType::kFloat16);

  // 将所有权重张量全部加载到一个 map 中（供模型使用）。
  std::unordered_map<std::string, Tensor> load_all(
      Context &ctx, DType target_dtype = DType::kFloat16);

 private:
  void parse_header();

  std::string path_;
  std::vector<char> file_bytes_;
  size_t data_offset_ = 0;
  std::unordered_map<std::string, TensorMeta> metas_;
  std::vector<std::string> keys_;
};

}  // namespace cudallm
