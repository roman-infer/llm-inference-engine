#include "cuda_llm/logger.h"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>

namespace cudallm {

const char *level_name(LogLevel level) {
  switch (level) {
    case LogLevel::kDebug:
      return "DEBUG";
    case LogLevel::kInfo:
      return "INFO ";
    case LogLevel::kWarn:
      return "WARN ";
    case LogLevel::kError:
      return "ERROR";
  }
  return "?????";
}

Logger &Logger::instance() {
  static Logger logger;
  return logger;
}

void Logger::log(LogLevel level, const char *file, int line,
                 const std::string &msg) {
  using namespace std::chrono;
  const auto now = system_clock::now();
  const auto t = system_clock::to_time_t(now);
  const auto us = duration_cast<microseconds>(now.time_since_epoch()) % 1000000;

  std::tm tm{};
#if defined(_WIN32)
  localtime_s(&tm, &t);
#else
  localtime_r(&t, &tm);
#endif

  // 仅提取文件名部分。
  const char *base = file;
  for (const char *p = file; *p; ++p) {
    if (*p == '/' || *p == '\\') base = p + 1;
  }

  static std::mutex io_mutex;
  std::lock_guard<std::mutex> lock(io_mutex);
  std::ostream &out = (level >= LogLevel::kWarn) ? std::cerr : std::cout;
  // 注意：setfill 是粘性状态，必须显式复位，否则会污染后续所有
  // std::setw（例如 bench_gemm 的表格被补成 '0'）。
  out << std::put_time(&tm, "%H:%M:%S") << '.' << std::setw(6)
      << std::setfill('0') << us.count() << std::setfill(' ') << ' '
      << level_name(level) << " [" << base << ':' << line << "] " << msg
      << std::endl;
}

}  // namespace cudallm
