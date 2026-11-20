// SPDX-License-Identifier: MIT
/**
 * @file      logging.h
 * @brief     Console logging and option checks.
 *
 * EquirectangularVIO note: uses stdio logging instead of spdlog so the ROS 2 node
 * does not clash with /usr/lib/libspdlog.so.1 (rcl_logging_spdlog) — that mix
 * caused free(): invalid pointer at rclcpp::init.
 */

#pragma once

#include <cstdio>
#include <string>
#include <type_traits>
#include <utility>

namespace vio {

inline void InitLogging() {}

// Minimal "{}" / "{:.Nf}" formatter (enough for OpenVIO log lines).
inline std::string FormatArgs(std::string format) { return format; }

template <typename T, typename... Args>
std::string FormatArgs(std::string format, T&& value, Args&&... args) {
  const auto pos = format.find('{');
  if (pos == std::string::npos) {
    return format;
  }
  const auto end = format.find('}', pos);
  if (end == std::string::npos) {
    return format;
  }
  const std::string spec = format.substr(pos, end - pos + 1);
  std::string out = format.substr(0, pos);
  using Decayed = std::decay_t<T>;
  if constexpr (std::is_same_v<Decayed, std::string> || std::is_same_v<Decayed, const char*> ||
                std::is_same_v<Decayed, char*>) {
    out += value;
  } else if constexpr (std::is_floating_point_v<Decayed>) {
    char buf[64];
    int precision = -1;
    if (spec.size() >= 5 && spec[1] == ':' && spec[2] == '.' && spec[spec.size() - 2] == 'f') {
      precision = 0;
      for (size_t i = 3; i + 2 < spec.size(); ++i) {
        if (spec[i] < '0' || spec[i] > '9') {
          precision = -1;
          break;
        }
        precision = precision * 10 + (spec[i] - '0');
      }
    }
    if (precision >= 0) {
      std::snprintf(buf, sizeof(buf), "%.*f", precision, static_cast<double>(value));
    } else {
      std::snprintf(buf, sizeof(buf), "%g", static_cast<double>(value));
    }
    out += buf;
  } else if constexpr (std::is_integral_v<Decayed>) {
    out += std::to_string(value);
  } else {
    out += spec;
  }
  out += FormatArgs(format.substr(end + 1), std::forward<Args>(args)...);
  return out;
}

template <typename... Args>
void LogInfo(Args&&... args) {
  const std::string line = FormatArgs(std::forward<Args>(args)...);
  std::fprintf(stdout, "[I] %s\n", line.c_str());
  std::fflush(stdout);
}

template <typename... Args>
void LogWarn(Args&&... args) {
  const std::string line = FormatArgs(std::forward<Args>(args)...);
  std::fprintf(stderr, "[W] %s\n", line.c_str());
  std::fflush(stderr);
}

template <typename... Args>
void LogError(Args&&... args) {
  const std::string line = FormatArgs(std::forward<Args>(args)...);
  std::fprintf(stderr, "[E] %s\n", line.c_str());
  std::fflush(stderr);
}

}  // namespace vio

#define LOG_INFO(...) ::vio::LogInfo(__VA_ARGS__)
#define LOG_WARN(...) ::vio::LogWarn(__VA_ARGS__)
#define LOG_ERROR(...) ::vio::LogError(__VA_ARGS__)

#define REQUIRE_OPTION(condition)                                               \
  do {                                                                          \
    if (!(condition)) {                                                         \
      LOG_ERROR("Invalid option: {} ({}:{})", #condition, __FILE__, __LINE__); \
      return false;                                                             \
    }                                                                           \
  } while (false)
