#pragma once

#include <sstream>
#include <string>

namespace cudallm {

enum class LogLevel { kDebug = 0, kInfo = 1, kWarn = 2, kError = 3 };

class Logger {
 public:
  static Logger &instance();

  void set_level(LogLevel level) { level_ = level; }
  LogLevel level() const { return level_; }

  void log(LogLevel level, const char *file, int line, const std::string &msg);

 private:
  Logger() = default;
  LogLevel level_ = LogLevel::kInfo;
};

const char *level_name(LogLevel level);

}  // namespace cudallm

// 用法：LOG_INFO("value=" << x);
// 注意：宏参数命名为 lvl（而非 level），以避免在展开时与
// Logger::level() 成员函数冲突。
#define LOG_AT(lvl, expr)                                                      \
  do {                                                                         \
    if (static_cast<int>(lvl) >=                                               \
        static_cast<int>(::cudallm::Logger::instance().level())) {             \
      std::ostringstream __oss;                                                \
      __oss << expr;                                                           \
      ::cudallm::Logger::instance().log(lvl, __FILE__, __LINE__,               \
                                        __oss.str());                          \
    }                                                                          \
  } while (0)

#define LOG_DEBUG(expr) LOG_AT(::cudallm::LogLevel::kDebug, expr)
#define LOG_INFO(expr) LOG_AT(::cudallm::LogLevel::kInfo, expr)
#define LOG_WARN(expr) LOG_AT(::cudallm::LogLevel::kWarn, expr)
#define LOG_ERROR(expr) LOG_AT(::cudallm::LogLevel::kError, expr)
