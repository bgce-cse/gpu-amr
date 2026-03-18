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

// -----------------------------------------------------------------------------
// Logging macros (compile-time stripped)
// -----------------------------------------------------------------------------
#if DEFAULT_SOURCE_LOG_LEVEL <= DEFAULT_SOURCE_LOG_LEVEL_TRACE
#    define DEFAULT_SOURCE_LOG_TRACE(fmt, ...)                   \
        utility::logging::default_logger::log(                   \
            spdlog::level::trace, fmt __VA_OPT__(, ) __VA_ARGS__ \
        )
#else
#    define DEFAULT_SOURCE_LOG_TRACE(...) DEFAULT_SOURCE_LOG_IGNORE(__VA_ARGS__)
#endif

#if DEFAULT_SOURCE_LOG_LEVEL <= DEFAULT_SOURCE_LOG_LEVEL_DEBUG
#    define DEFAULT_SOURCE_LOG_DEBUG(fmt, ...)                   \
        utility::logging::default_logger::log(                   \
            spdlog::level::debug, fmt __VA_OPT__(, ) __VA_ARGS__ \
        )
#else
#    define DEFAULT_SOURCE_LOG_DEBUG(...) DEFAULT_SOURCE_LOG_IGNORE(__VA_ARGS__)
#endif

#if DEFAULT_SOURCE_LOG_LEVEL <= DEFAULT_SOURCE_LOG_LEVEL_INFO
#    define DEFAULT_SOURCE_LOG_INFO(fmt, ...)                   \
        utility::logging::default_logger::log(                  \
            spdlog::level::info, fmt __VA_OPT__(, ) __VA_ARGS__ \
        )
#else
#    define DEFAULT_SOURCE_LOG_INFO(...) DEFAULT_SOURCE_LOG_IGNORE(__VA_ARGS__)
#endif

#if DEFAULT_SOURCE_LOG_LEVEL <= DEFAULT_SOURCE_LOG_LEVEL_PROGRESS
#    define DEFAULT_SOURCE_LOG_PROGRESS(fmt, ...)               \
        utility::logging::default_logger::log(                  \
            spdlog::level::info, fmt __VA_OPT__(, ) __VA_ARGS__ \
        )
#else
#    define DEFAULT_SOURCE_LOG_PROGRESS(...) DEFAULT_SOURCE_LOG_IGNORE(__VA_ARGS__)
#endif

#if DEFAULT_SOURCE_LOG_LEVEL <= DEFAULT_SOURCE_LOG_LEVEL_WARNING
#    define DEFAULT_SOURCE_LOG_WARNING(fmt, ...)                \
        utility::logging::default_logger::log(                  \
            spdlog::level::warn, fmt __VA_OPT__(, ) __VA_ARGS__ \
        )
#else
#    define DEFAULT_SOURCE_LOG_WARNING(...) DEFAULT_SOURCE_LOG_IGNORE(__VA_ARGS__)
#endif

#if DEFAULT_SOURCE_LOG_LEVEL <= DEFAULT_SOURCE_LOG_LEVEL_ERROR
#    define DEFAULT_SOURCE_LOG_ERROR(fmt, ...)                 \
        utility::logging::default_logger::log(                 \
            spdlog::level::err, fmt __VA_OPT__(, ) __VA_ARGS__ \
        )
#else
#    define DEFAULT_SOURCE_LOG_ERROR(...) DEFAULT_SOURCE_LOG_IGNORE(__VA_ARGS__)
#endif

#if DEFAULT_SOURCE_LOG_LEVEL <= DEFAULT_SOURCE_LOG_LEVEL_FATAL
#    define DEFAULT_SOURCE_LOG_FATAL(fmt, ...)                      \
        utility::logging::default_logger::log(                      \
            spdlog::level::critical, fmt __VA_OPT__(, ) __VA_ARGS__ \
        )
#else
#    define DEFAULT_SOURCE_LOG_FATAL(...) DEFAULT_SOURCE_LOG_IGNORE(__VA_ARGS__)
#endif

// Avoid includes and definitions if logging is off
#if DEFAULT_SOURCE_LOG_LEVEL < UTILITY_DEFAULT_LOG_LEVEL_OFF

#    include <fmt/core.h>
#    include <fmt/format.h>
#    include <spdlog/common.h>
#    include <spdlog/spdlog.h>
#    include <string_view>

namespace utility::logging
{

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
        log(spdlog::level::level_enum   sev,
            fmt::format_string<Args...> fmt,
            Args const&... args) -> void
    {
        const auto msg = detail::format_to_buffer(fmt, fmt::make_format_args(args...));
        log(sev, msg);
    }

    static auto log(spdlog::level::level_enum sev, std::string_view msg) -> void;
};

} // namespace utility::logging

#endif

#endif // INCLUDED_UTILITY_LOGGING
