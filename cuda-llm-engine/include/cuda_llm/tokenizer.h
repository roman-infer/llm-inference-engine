#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cudallm {

// GPT-2 风格的字节级 BPE 分词器。
//
//   原始文本 --(预分词正则)--> 单词字符串
//   单词 --(bytes_to_unicode)--> 符号序列 --(BPE 合并)--> token id
//
// 预分词器是一个紧凑的手写扫描器，对于 ASCII 和常见 UTF-8 文本，
// 等价于 GPT-2 的正则：
//   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+
class BPETokenizer {
 public:
  BPETokenizer() = default;

  static BPETokenizer from_dir(const std::string &directory);
  static BPETokenizer from_files(const std::string &vocab_json_path,
                                 const std::string &merges_path);

  std::vector<int> encode(const std::string &text) const;
  std::string decode(const std::vector<int> &ids) const;

  int vocab_size() const { return static_cast<int>(id_to_token_.size()); }
  int eos_id() const { return eos_id_; }
  int bos_id() const { return bos_id_; }

 private:
  void build_byte_tables();
  void load_vocab(const std::string &vocab_json_path);
  void load_merges(const std::string &merges_path);

  std::vector<std::string> pretokenize(const std::string &text) const;
  std::vector<int> bpe_encode_word(const std::string &word) const;

  // 解码字节偏移 i 处的一个 UTF-8 码点；返回码点值并将 i 前移。
  static uint32_t decode_utf8(const std::string &s, size_t &i);
  static void append_codepoint_utf8(std::string &out, uint32_t code);

  std::unordered_map<std::string, int> token_to_id_;
  std::vector<std::string> id_to_token_;
  std::map<std::pair<std::string, std::string>, int> merge_ranks_;

  // 字节 (0..255) <-> 映射符号（GPT-2 unicode 字符的 UTF-8 字符串）
  std::vector<std::string> byte_to_symbol_;
  std::unordered_map<std::string, int> symbol_to_byte_;

  int eos_id_ = 50256;
  int bos_id_ = 50256;
};

}  // namespace cudallm
