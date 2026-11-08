#include "cuda_llm/tokenizer.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "cuda_llm/json_min.h"

namespace cudallm {

// ---------------------------------------------------------------------------
// UTF-8 辅助函数
// ---------------------------------------------------------------------------
uint32_t BPETokenizer::decode_utf8(const std::string &s, size_t &i) {
  unsigned char c = static_cast<unsigned char>(s[i]);
  uint32_t code = 0;
  int extra = 0;
  if (c < 0x80) {
    code = c;
    extra = 0;
  } else if ((c & 0xE0) == 0xC0) {
    code = c & 0x1F;
    extra = 1;
  } else if ((c & 0xF0) == 0xE0) {
    code = c & 0x0F;
    extra = 2;
  } else if ((c & 0xF8) == 0xF0) {
    code = c & 0x07;
    extra = 3;
  } else {
    throw std::runtime_error("UTF-8: bad lead byte");
  }
  ++i;
  for (int k = 0; k < extra; ++k) {
    code = (code << 6) | (static_cast<unsigned char>(s[i]) & 0x3F);
    ++i;
  }
  return code;
}

void BPETokenizer::append_codepoint_utf8(std::string &out, uint32_t code) {
  if (code < 0x80) {
    out.push_back(static_cast<char>(code));
  } else if (code < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (code >> 6)));
    out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
  } else if (code < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (code >> 12)));
    out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (code >> 18)));
    out.push_back(static_cast<char>(0x80 | ((code >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
  }
}

// ---------------------------------------------------------------------------
// 字符分类（Unicode 类别的紧凑替代）
// ---------------------------------------------------------------------------
namespace {

bool is_space(uint32_t c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == 0x0B ||
         c == 0x0C || c == 0x85 || c == 0x2028 || c == 0x2029;
}
bool is_letter(uint32_t c) {
  if (c < 0x80)
    return std::isalpha(static_cast<int>(c)) != 0;
  return true;  // 带重音的拉丁字母、CJK 等视为字母
}
bool is_number(uint32_t c) {
  if (c < 0x80) return std::isdigit(static_cast<int>(c)) != 0;
  return c == 0xB2 || c == 0xB3 || c == 0xB9 ||
         (c >= 0xBC && c <= 0xBE);
}

}  // namespace

// ---------------------------------------------------------------------------
// 字节 <-> unicode 符号表（GPT-2 的 bytes_to_unicode）
// ---------------------------------------------------------------------------
void BPETokenizer::build_byte_tables() {
  byte_to_symbol_.resize(256);
  symbol_to_byte_.clear();

  auto printable = [](int b) {
    return (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) ||
           (b >= 0xAE && b <= 0xFF);
  };

  int n = 0;
  for (int b = 0; b < 256; ++b) {
    uint32_t code = printable(b) ? static_cast<uint32_t>(b)
                                 : static_cast<uint32_t>(256 + n++);
    std::string sym;
    append_codepoint_utf8(sym, code);
    byte_to_symbol_[b] = sym;
    symbol_to_byte_[sym] = b;
  }
}

// ---------------------------------------------------------------------------
// 文件加载
// ---------------------------------------------------------------------------
BPETokenizer BPETokenizer::from_dir(const std::string &directory) {
  const std::string sep =
      directory.find('/') != std::string::npos ? "/" : "\\";
  return from_files(directory + sep + "vocab.json",
                    directory + sep + "merges.txt");
}

BPETokenizer BPETokenizer::from_files(const std::string &vocab_path,
                                      const std::string &merges_path) {
  BPETokenizer tok;
  tok.build_byte_tables();
  tok.load_vocab(vocab_path);
  tok.load_merges(merges_path);
  return tok;
}

void BPETokenizer::load_vocab(const std::string &vocab_path) {
  std::ifstream f(vocab_path, std::ios::binary);
  if (!f) throw std::runtime_error("Cannot open vocab: " + vocab_path);
  std::stringstream ss;
  ss << f.rdbuf();
  JsonValue root = JsonParser::parse(ss.str());

  id_to_token_.resize(root.obj.size());
  for (const auto &kv : root.obj) {
    int id = kv.second.as_int();
    id_to_token_[id] = kv.first;
    token_to_id_[kv.first] = id;
  }
  auto eos = token_to_id_.find("<|endoftext|>");
  if (eos != token_to_id_.end()) eos_id_ = bos_id_ = eos->second;
}

void BPETokenizer::load_merges(const std::string &merges_path) {
  std::ifstream f(merges_path);
  if (!f) throw std::runtime_error("Cannot open merges: " + merges_path);
  std::string line;
  int rank = 0;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    const size_t sp = line.find(' ');
    if (sp == std::string::npos) continue;
    std::pair<std::string, std::string> pair = {
        line.substr(0, sp), line.substr(sp + 1)};
    merge_ranks_[pair] = rank++;
  }
}

// ---------------------------------------------------------------------------
// 预分词
// ---------------------------------------------------------------------------
std::vector<std::string> BPETokenizer::pretokenize(
    const std::string &text) const {
  std::vector<std::string> tokens;
  const size_t n = text.size();
  size_t i = 0;

  auto starts_with = [&](const char *lit) {
    size_t len = std::strlen(lit);
    return i + len <= n && text.compare(i, len, lit) == 0;
  };

  while (i < n) {
    const size_t save = i;
    const uint32_t c = decode_utf8(text, i);

    // 1) 缩写（最长匹配优先）
    if (c == '\'') {
      static const char *contractions[] = {"'re", "'ve", "'ll",
                                           "'s",  "'t",  "'m",
                                           "'d"};
      bool matched = false;
      for (const char *p : contractions) {
        if (starts_with(p)) {
          tokens.emplace_back(p);
          i = save + std::strlen(p);
          matched = true;
          break;
        }
      }
      if (matched) continue;
    }

    // 2) 空白字符序列
    if (is_space(c)) {
      size_t j = i;  // i 已跳过第一个字符
      while (j < n) {
        size_t k = j;
        uint32_t d = decode_utf8(text, k);
        if (!is_space(d)) break;
        j = k;
      }
      if (j == n || j != save + 1) {
        // 尾部空格，或长度超过一个空格的连续空白 -> 作为纯空格 token
        tokens.push_back(text.substr(save, j - save));
        i = j;
        continue;
      }
      // 恰好一个空格后紧跟内容：将其附加到下一个 token
      // （不做处理，回到 'save' 处重新分类）
      i = save;
    }

    const size_t token_start = save;
    size_t cur = i;

    // 消费一个可选的前导空格。
    size_t body_start = cur;
    {
      size_t probe = cur;
      uint32_t d = decode_utf8(text, probe);
      if (d == ' ') body_start = probe;
    }

    size_t probe = body_start;
    uint32_t first = decode_utf8(text, probe);

    auto category_run = [&](bool (*pred)(uint32_t)) {
      size_t k = probe;
      while (k < n) {
        size_t nxt = k;
        uint32_t d = decode_utf8(text, nxt);
        if (!pred(d)) break;
        k = nxt;
      }
      return k;
    };

    size_t end;
    if (is_letter(first)) {
      end = category_run(is_letter);
    } else if (is_number(first)) {
      end = category_run(is_number);
    } else if (!is_space(first)) {
      end = category_run([](uint32_t d) {
        return !is_space(d) && !is_letter(d) && !is_number(d);
      });
    } else {
      end = probe;  // 防御性处理
    }

    tokens.push_back(text.substr(token_start, end - token_start));
    i = end;
  }

  return tokens;
}

// ---------------------------------------------------------------------------
// BPE 编码
// ---------------------------------------------------------------------------
std::vector<int> BPETokenizer::bpe_encode_word(
    const std::string &word) const {
  std::vector<std::string> sym;
  sym.reserve(word.size());
  for (unsigned char b : word) {
    sym.push_back(byte_to_symbol_[b]);
  }

  while (sym.size() >= 2) {
    int best_rank = std::numeric_limits<int>::max();
    size_t best_idx = 0;
    for (size_t p = 0; p + 1 < sym.size(); ++p) {
      auto it = merge_ranks_.find({sym[p], sym[p + 1]});
      if (it != merge_ranks_.end() && it->second < best_rank) {
        best_rank = it->second;
        best_idx = p;
      }
    }
    if (best_rank == std::numeric_limits<int>::max()) break;

    sym[best_idx] += sym[best_idx + 1];
    sym.erase(sym.begin() + best_idx + 1);
  }

  std::vector<int> ids;
  ids.reserve(sym.size());
  for (const std::string &s : sym) {
    auto it = token_to_id_.find(s);
    if (it == token_to_id_.end()) {
      throw std::runtime_error("BPE: symbol not in vocab: " + s);
    }
    ids.push_back(it->second);
  }
  return ids;
}

std::vector<int> BPETokenizer::encode(const std::string &text) const {
  std::vector<int> ids;
  for (const std::string &word : pretokenize(text)) {
    std::vector<int> w = bpe_encode_word(word);
    ids.insert(ids.end(), w.begin(), w.end());
  }
  return ids;
}

std::string BPETokenizer::decode(const std::vector<int> &ids) const {
  std::string mapped;
  for (int id : ids) {
    if (id >= 0 && id < static_cast<int>(id_to_token_.size())) {
      mapped += id_to_token_[id];
    }
  }
  std::string raw_bytes;
  size_t i = 0;
  while (i < mapped.size()) {
    std::string sym;
    uint32_t code = decode_utf8(mapped, i);
    append_codepoint_utf8(sym, code);
    auto it = symbol_to_byte_.find(sym);
    if (it != symbol_to_byte_.end()) {
      raw_bytes.push_back(static_cast<char>(it->second));
    }
  }
  return raw_bytes;
}

}  // namespace cudallm
