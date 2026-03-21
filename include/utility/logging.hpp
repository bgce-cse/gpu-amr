#ifndef INCLUDED_UTILITY_LOGGING
#define INCLUDED_UTILITY_LOGGING

#include "config/definitions.hpp"
#include "utility/macro_definitions.hpp"

#ifndef UTILITY_DEFAULT_LOG_LEVEL_TRACE
#    define UTILITY_DEFAULT_LOG_LEVEL_TRACE 1
#endif
#ifndef UTILITY_DEFAULT_LOG_LEVEL_DEBUG
#    define UTILITY_DEFAULT_LOG_LEVEL_DEBUG 2
#endif
#ifndef UTILITY_DEFAULT_LOG_LEVEL_INFO
#    define UTILITY_DEFAULT_LOG_LEVEL_INFO 3
#endif
#ifndef UTILITY_DEFAULT_LOG_LEVEL_PROGRESS
#    define UTILITY_DEFAULT_LOG_LEVEL_PROGRESS 4
#endif
#ifndef UTILITY_DEFAULT_LOG_LEVEL_WARNING
#    define UTILITY_DEFAULT_LOG_LEVEL_WARNING 5
#endif
#ifndef UTILITY_DEFAULT_LOG_LEVEL_ERROR
#    define UTILITY_DEFAULT_LOG_LEVEL_ERROR 6
#endif
#ifndef UTILITY_DEFAULT_LOG_LEVEL_FATAL
#    define UTILITY_DEFAULT_LOG_LEVEL_FATAL 7
#endif
#ifndef UTILITY_DEFAULT_LOG_LEVEL_OFF
#    define UTILITY_DEFAULT_LOG_LEVEL_OFF 8
#endif

#define DEFAULT_SOURCE_LOG_LEVEL_TRACE    UTILITY_DEFAULT_LOG_LEVEL_TRACE
#define DEFAULT_SOURCE_LOG_LEVEL_DEBUG    UTILITY_DEFAULT_LOG_LEVEL_DEBUG
#define DEFAULT_SOURCE_LOG_LEVEL_INFO     UTILITY_DEFAULT_LOG_LEVEL_INFO
#define DEFAULT_SOURCE_LOG_LEVEL_PROGRESS UTILITY_DEFAULT_LOG_LEVEL_PROGRESS
#define DEFAULT_SOURCE_LOG_LEVEL_WARNING  UTILITY_DEFAULT_LOG_LEVEL_WARNING
#define DEFAULT_SOURCE_LOG_LEVEL_ERROR    UTILITY_DEFAULT_LOG_LEVEL_ERROR
#define DEFAULT_SOURCE_LOG_LEVEL_FATAL    UTILITY_DEFAULT_LOG_LEVEL_FATAL
#define DEFAULT_SOURCE_LOG_LEVEL_OFF      UTILITY_DEFAULT_LOG_LEVEL_OFF

#ifdef UTILITY_LOG_LEVEL
#    define DEFAULT_SOURCE_LOG_LEVEL \
        UTILITY_CONCATENATE_MACRO(DEFAULT_SOURCE_LOG_LEVEL_, UTILITY_LOG_LEVEL)
#else
#    define DEFAULT_SOURCE_LOG_LEVEL DEFAULT_SOURCE_LOG_LEVEL_INFO
#endif

#define DEFAULT_SOURCE_LOG_IGNORE(...) ((void)0)

#ifdef ENABLE_SPDLOG
#    define LOG_LEVEL_TRACE    spdlog::level::trace
#    define LOG_LEVEL_DEBUG    spdlog::level::debug
#    define LOG_LEVEL_INFO     spdlog::level::info
#    define LOG_LEVEL_PROGRESS spdlog::level::info
#    define LOG_LEVEL_WARNING  spdlog::level::warn
#    define LOG_LEVEL_ERROR    spdlog::level::err
#    define LOG_LEVEL_FATAL    spdlog::level::critical
#else
enum struct log_level
{
    trace,
    debug,
    info,
    progress,
    warning,
    error,
    fatal,
};
#    define LOG_LEVEL_TRACE    log_level::trace
#    define LOG_LEVEL_DEBUG    log_level::debug
#    define LOG_LEVEL_INFO     log_level::info
#    define LOG_LEVEL_PROGRESS log_level::progress
#    define LOG_LEVEL_WARNING  log_level::warning
#    define LOG_LEVEL_ERROR    log_level::error
#    define LOG_LEVEL_FATAL    log_level::fatal
#endif

// -----------------------------------------------------------------------------
// Logging macros (compile-time stripped)
// -----------------------------------------------------------------------------
#if DEFAULT_SOURCE_LOG_LEVEL <= DEFAULT_SOURCE_LOG_LEVEL_TRACE
#    define DEFAULT_SOURCE_LOG_TRACE(fmt, ...)              \
        utility::logging::default_logger::log(              \
            LOG_LEVEL_TRACE, fmt __VA_OPT__(, ) __VA_ARGS__ \
        )
#else
#    define DEFAULT_SOURCE_LOG_TRACE(...) DEFAULT_SOURCE_LOG_IGNORE(__VA_ARGS__)
#endif

#if DEFAULT_SOURCE_LOG_LEVEL <= DEFAULT_SOURCE_LOG_LEVEL_DEBUG
#    define DEFAULT_SOURCE_LOG_DEBUG(fmt, ...)              \
        utility::logging::default_logger::log(              \
            LOG_LEVEL_DEBUG, fmt __VA_OPT__(, ) __VA_ARGS__ \
        )
#else
#    define DEFAULT_SOURCE_LOG_DEBUG(...) DEFAULT_SOURCE_LOG_IGNORE(__VA_ARGS__)
#endif

#if DEFAULT_SOURCE_LOG_LEVEL <= DEFAULT_SOURCE_LOG_LEVEL_INFO
#    define DEFAULT_SOURCE_LOG_INFO(fmt, ...)              \
        utility::logging::default_logger::log(             \
            LOG_LEVEL_INFO, fmt __VA_OPT__(, ) __VA_ARGS__ \
        )
#else
#    define DEFAULT_SOURCE_LOG_INFO(...) DEFAULT_SOURCE_LOG_IGNORE(__VA_ARGS__)
#endif

#if DEFAULT_SOURCE_LOG_LEVEL <= DEFAULT_SOURCE_LOG_LEVEL_PROGRESS
#    define DEFAULT_SOURCE_LOG_PROGRESS(fmt, ...)              \
        utility::logging::default_logger::log(                 \
            LOG_LEVEL_PROGRESS, fmt __VA_OPT__(, ) __VA_ARGS__ \
        )
#else
#    define DEFAULT_SOURCE_LOG_PROGRESS(...) DEFAULT_SOURCE_LOG_IGNORE(__VA_ARGS__)
#endif

#if DEFAULT_SOURCE_LOG_LEVEL <= DEFAULT_SOURCE_LOG_LEVEL_WARNING
#    define DEFAULT_SOURCE_LOG_WARNING(fmt, ...)              \
        utility::logging::default_logger::log(                \
            LOG_LEVEL_WARNING, fmt __VA_OPT__(, ) __VA_ARGS__ \
        )
#else
#    define DEFAULT_SOURCE_LOG_WARNING(...) DEFAULT_SOURCE_LOG_IGNORE(__VA_ARGS__)
#endif

#if DEFAULT_SOURCE_LOG_LEVEL <= DEFAULT_SOURCE_LOG_LEVEL_ERROR
#    define DEFAULT_SOURCE_LOG_ERROR(fmt, ...)              \
        utility::logging::default_logger::log(              \
            LOG_LEVEL_ERROR, fmt __VA_OPT__(, ) __VA_ARGS__ \
        )
#else
#    define DEFAULT_SOURCE_LOG_ERROR(...) DEFAULT_SOURCE_LOG_IGNORE(__VA_ARGS__)
#endif

#if DEFAULT_SOURCE_LOG_LEVEL <= DEFAULT_SOURCE_LOG_LEVEL_FATAL
#    define DEFAULT_SOURCE_LOG_FATAL(fmt, ...)              \
        utility::logging::default_logger::log(              \
            LOG_LEVEL_FATAL, fmt __VA_OPT__(, ) __VA_ARGS__ \
        )
#else
#    define DEFAULT_SOURCE_LOG_FATAL(...) DEFAULT_SOURCE_LOG_IGNORE(__VA_ARGS__)
#endif

// Avoid includes and definitions if logging is off
#if DEFAULT_SOURCE_LOG_LEVEL < UTILITY_DEFAULT_LOG_LEVEL_OFF

#    include <string_view>
#    ifdef ENABLE_SPDLOG
#        include <fmt/core.h>
#        include <fmt/format.h>
#        include <fmt/ranges.h>
#        include <spdlog/common.h>
#        include <spdlog/spdlog.h>
#    else
#        include <format>
#        include <iostream>
#    endif

namespace utility::logging
{

#    ifdef ENABLE_SPDLOG
namespace detail
{

inline std::string_view format_to_buffer(fmt::string_view fmt, fmt::format_args args)
{
    thread_local fmt::memory_buffer buffer;
    buffer.clear();

    fmt::vformat_to(std::back_inserter(buffer), fmt, args);

    return { buffer.data(), buffer.size() };
}

} // namespace detail

class default_logger
{
private:
    static auto default_source() -> std::shared_ptr<spdlog::logger>&
    {
        static auto log_source = spdlog::get("default_source");
        return log_source;
    }

    static auto default_async_source() -> std::shared_ptr<spdlog::logger>&
    {
        static auto log_source = spdlog::get("default_async_source");
        return log_source;
    }

public:
    static auto init() -> void;

    template <typename... Args>
    static auto
        log(spdlog::level::level_enum const sev,
            fmt::format_string<Args...>     fmt,
            Args const&... args) -> void
    {
        const auto msg =
            detail::format_to_buffer(fmt.get(), fmt::make_format_args(args...));
        log(sev, msg);
    }

    static auto log(spdlog::level::level_enum sev, std::string_view msg) -> void;
};

#    else

struct default_logger
{
    template <typename... Args>
    static void log(log_level, std::string_view fmt, Args&&... args)
    {
        std::cout << std::vformat(fmt, std::make_format_args(args...)) << '\n';
    }

    static void log(log_level, std::string_view msg)
    {
        std::cout << msg << '\n';
    }
};

#    endif // ENABLE_SPDLOG

} // namespace utility::logging

#endif

#endif // INCLUDED_UTILITY_LOGGING
