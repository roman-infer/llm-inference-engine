#pragma once

// 最小化递归下降 JSON 解析器，足以解析 safetensors 头部
// 和简单配置文件。零依赖（不使用 nlohmann/json）。

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace cudallm {

struct JsonValue {
  enum Type { kNull, kBool, kNumber, kString, kArray, kObject } type = kNull;

  bool boolean = false;
  double number = 0.0;
  std::string str;
  std::vector<JsonValue> arr;
  std::map<std::string, JsonValue> obj;

  bool contains(const std::string &key) const {
    return obj.find(key) != obj.end();
  }
  const JsonValue &at(const std::string &key) const { return obj.at(key); }

  double as_number() const { return number; }
  int as_int() const { return static_cast<int>(number); }
  int64_t as_i64() const { return static_cast<int64_t>(number); }
  const std::string &as_string() const { return str; }
  const std::vector<JsonValue> &as_array() const { return arr; }
};

class JsonParser {
 public:
  static JsonValue parse(const std::string &text) {
    JsonParser parser(text);
    parser.skip_ws();
    JsonValue v = parser.parse_value();
    parser.skip_ws();
    if (parser.pos_ != text.size()) {
      throw std::runtime_error("JSON: trailing characters");
    }
    return v;
  }

 private:
  explicit JsonParser(const std::string &text) : text_(text) {}

  char peek() const { return text_[pos_]; }
  char consume() { return text_[pos_++]; }
  void expect(char c) {
    if (pos_ >= text_.size() || text_[pos_] != c) {
      throw std::runtime_error(std::string("JSON: expected '") + c + "'");
    }
    ++pos_;
  }

  void skip_ws() {
    while (pos_ < text_.size() &&
           (text_[pos_] == ' ' || text_[pos_] == '\t' ||
            text_[pos_] == '\n' || text_[pos_] == '\r')) {
      ++pos_;
    }
  }

  JsonValue parse_value() {
    skip_ws();
    if (pos_ >= text_.size()) throw std::runtime_error("JSON: unexpected end");
    char c = peek();
    if (c == '{') return parse_object();
    if (c == '[') return parse_array();
    if (c == '"') {
      JsonValue v;
      v.type = JsonValue::kString;
      v.str = parse_string();
      return v;
    }
    if (c == 't' || c == 'f') return parse_bool();
    if (c == 'n') return parse_null();
    return parse_number();
  }

  JsonValue parse_object() {
    JsonValue v;
    v.type = JsonValue::kObject;
    expect('{');
    skip_ws();
    if (peek() == '}') {
      consume();
      return v;
    }
    while (true) {
      skip_ws();
      std::string key = parse_string();
      skip_ws();
      expect(':');
      JsonValue val = parse_value();
      v.obj.emplace(std::move(key), std::move(val));
      skip_ws();
      char c = consume();
      if (c == '}') break;
      if (c != ',') throw std::runtime_error("JSON: expected ',' or '}'");
    }
    return v;
  }

  JsonValue parse_array() {
    JsonValue v;
    v.type = JsonValue::kArray;
    expect('[');
    skip_ws();
    if (peek() == ']') {
      consume();
      return v;
    }
    while (true) {
      v.arr.push_back(parse_value());
      skip_ws();
      char c = consume();
      if (c == ']') break;
      if (c != ',') throw std::runtime_error("JSON: expected ',' or ']'");
    }
    return v;
  }

  std::string parse_string() {
    expect('"');
    std::string out;
    while (true) {
      if (pos_ >= text_.size()) throw std::runtime_error("JSON: bad string");
      char c = consume();
      if (c == '"') break;
      if (c == '\\') {
        char e = consume();
        switch (e) {
          case '"':
            out.push_back('"');
            break;
          case '\\':
            out.push_back('\\');
            break;
          case '/':
            out.push_back('/');
            break;
          case 'n':
            out.push_back('\n');
            break;
          case 't':
            out.push_back('\t');
            break;
          case 'r':
            out.push_back('\r');
            break;
          case 'b':
            out.push_back('\b');
            break;
          case 'f':
            out.push_back('\f');
            break;
          case 'u': {
            if (pos_ + 4 > text_.size())
              throw std::runtime_error("JSON: bad unicode escape");
            unsigned code = 0;
            for (int i = 0; i < 4; ++i) {
              char h = consume();
              code <<= 4;
              if (h >= '0' && h <= '9')
                code |= h - '0';
              else if (h >= 'a' && h <= 'f')
                code |= h - 'a' + 10;
              else if (h >= 'A' && h <= 'F')
                code |= h - 'A' + 10;
              else
                throw std::runtime_error("JSON: bad hex digit");
            }
            append_utf8(out, code);
            break;
          }
          default:
            throw std::runtime_error("JSON: bad escape");
        }
      } else {
        out.push_back(c);
      }
    }
    return out;
  }

  static void append_utf8(std::string &out, unsigned code) {
    if (code < 0x80) {
      out.push_back(static_cast<char>(code));
    } else if (code < 0x800) {
      out.push_back(static_cast<char>(0xC0 | (code >> 6)));
      out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
    } else {
      out.push_back(static_cast<char>(0xE0 | (code >> 12)));
      out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
    }
  }

  JsonValue parse_number() {
    size_t start = pos_;
    if (peek() == '-') consume();
    while (pos_ < text_.size() &&
           (std::isdigit(static_cast<unsigned char>(peek())) || peek() == '.' ||
            peek() == 'e' || peek() == 'E' || peek() == '+' || peek() == '-')) {
      consume();
    }
    JsonValue v;
    v.type = JsonValue::kNumber;
    v.number = std::strtod(text_.c_str() + start, nullptr);
    return v;
  }

  JsonValue parse_bool() {
    if (text_.compare(pos_, 4, "true") == 0) {
      pos_ += 4;
      JsonValue v;
      v.type = JsonValue::kBool;
      v.boolean = true;
      return v;
    }
    if (text_.compare(pos_, 5, "false") == 0) {
      pos_ += 5;
      JsonValue v;
      v.type = JsonValue::kBool;
      v.boolean = false;
      return v;
    }
    throw std::runtime_error("JSON: bad literal");
  }

  JsonValue parse_null() {
    if (text_.compare(pos_, 4, "null") == 0) {
      pos_ += 4;
      return JsonValue{};
    }
    throw std::runtime_error("JSON: bad literal");
  }

  const std::string &text_;
  size_t pos_ = 0;
};

}  // namespace cudallm
