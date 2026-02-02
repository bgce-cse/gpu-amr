#ifndef INCLUDED_UTILITY_LOGGING
#define INCLUDED_UTILITY_LOGGING

#include "macro_definitions.hpp"

#ifndef AMR_DEFAULT_LOG_LEVEL_TRACE
#    define AMR_DEFAULT_LOG_LEVEL_TRACE 1
#endif
#ifndef AMR_DEFAULT_LOG_LEVEL_DEBUG
#    define AMR_DEFAULT_LOG_LEVEL_DEBUG 2
#endif
#ifndef AMR_DEFAULT_LOG_LEVEL_INFO
#    define AMR_DEFAULT_LOG_LEVEL_INFO 3
#endif
#ifndef AMR_DEFAULT_LOG_LEVEL_WARNING
#    define AMR_DEFAULT_LOG_LEVEL_WARNING 4
#endif
#ifndef AMR_DEFAULT_LOG_LEVEL_ERROR
#    define AMR_DEFAULT_LOG_LEVEL_ERROR 5
#endif
#ifndef AMR_DEFAULT_LOG_LEVEL_FATAL
#    define AMR_DEFAULT_LOG_LEVEL_FATAL 6
#endif
#ifndef AMR_DEFAULT_LOG_LEVEL_OFF
#    define AMR_DEFAULT_LOG_LEVEL_OFF 7
#endif

#define DEFAULT_SOURCE_LOG_LEVEL_TRACE   AMR_DEFAULT_LOG_LEVEL_TRACE
#define DEFAULT_SOURCE_LOG_LEVEL_DEBUG   AMR_DEFAULT_LOG_LEVEL_DEBUG
#define DEFAULT_SOURCE_LOG_LEVEL_INFO    AMR_DEFAULT_LOG_LEVEL_INFO
#define DEFAULT_SOURCE_LOG_LEVEL_WARNING AMR_DEFAULT_LOG_LEVEL_WARNING
#define DEFAULT_SOURCE_LOG_LEVEL_ERROR   AMR_DEFAULT_LOG_LEVEL_ERROR
#define DEFAULT_SOURCE_LOG_LEVEL_FATAL   AMR_DEFAULT_LOG_LEVEL_FATAL
#define DEFAULT_SOURCE_LOG_LEVEL_OFF     AMR_DEFAULT_LOG_LEVEL_OFF

#ifdef AMR_LOG_LEVEL
#    define DEFAULT_SOURCE_LOG_LEVEL \
        UTILITY_CONCATENATE_MACRO(DEFAULT_SOURCE_LOG_LEVEL_, AMR_LOG_LEVEL)
#else
#    define DEFAULT_SOURCE_LOG_LEVEL DEFAULT_SOURCE_LOG_LEVEL_INFO
#endif

#define DEFAULT_SOURCE_LOG_IGNORE(...) ((void)0)

//
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
#if DEFAULT_SOURCE_LOG_LEVEL < AMR_DEFAULT_LOG_LEVEL_OFF

#    include "spdlog/async.h"
#    include "spdlog/sinks/rotating_file_sink.h"
#    include <format>
#    include <memory>
#    include <spdlog/common.h>
#    include <spdlog/sinks/basic_file_sink.h>
#    include <spdlog/sinks/stdout_color_sinks.h>
#    include <spdlog/spdlog.h>
#    include <string_view>
#    include <thread>
#    include <unistd.h>
#    include <vector>

namespace utility::logging
{

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
    static auto init() -> void
    {
        static bool initialized = []
        {
            spdlog::init_thread_pool(8192, 1);

            auto stdout_sink  = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
            auto general_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(
                "logs/general.log", true
            );
            auto rotating_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
                "logs/backtrace.log", 1024 * 1024 * 10, 3
            );
            auto errors_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(
                "logs/errors.log", true
            );
            errors_sink->set_level(spdlog::level::warn);

            std::vector<spdlog::sink_ptr> async_sinks{ general_sink,
                                                       rotating_sink,
                                                       errors_sink };

            auto logger = std::make_shared<spdlog::logger>("default_source", stdout_sink);
            logger->set_pattern("[%H:%M:%S.%e] [%^%l%$] %v");
            logger->set_level(spdlog::level::info);
            logger->flush_on(spdlog::level::warn);

            auto async_logger = std::make_shared<spdlog::async_logger>(
                "default_async_source",
                async_sinks.begin(),
                async_sinks.end(),
                spdlog::thread_pool(),
                spdlog::async_overflow_policy::overrun_oldest
            );
            async_logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
            async_logger->set_level(spdlog::level::trace);

            spdlog::register_logger(logger);
            spdlog::register_logger(async_logger);

            // Initial messages
            async_logger->info("Build:");
            async_logger->info("  type      : {}", AMR_BUILD_TYPE);
            async_logger->info(
                "  compiler  : {} {}", AMR_COMPILER_ID, AMR_COMPILER_VERSION
            );
            async_logger->info("  git hash  : {}", AMR_GIT_HASH);
            async_logger->info("Runtime configuration:");
            async_logger->info(
                "  log level : {}", spdlog::level::to_string_view(logger->level())
            );
            async_logger->info("  threads   : {}", std::thread::hardware_concurrency());
            async_logger->info("  pid       : {}", ::getpid());

            return true;
        }();

        (void)initialized;
    }

    static auto log(spdlog::level::level_enum sev, std::string_view msg) -> void
    {
        init();
        dispatch(sev, msg);
    }

    template <typename... Args>
    static auto
        log(spdlog::level::level_enum   sev,
            std::format_string<Args...> fmt,
            Args&&... args) -> void
    {
        init();
        dispatch(sev, std::format(fmt, std::forward<Args>(args)...));
    }

    static auto dispatch(spdlog::level::level_enum sev, std::string_view msg) -> void
    {
        if (sev >= spdlog::level::info)
        {
            default_source()->log(sev, msg);
        }
        default_async_source()->log(sev, msg);
    }
};

} // namespace utility::logging
#endif

#endif // INCLUDED_UTILITY_LOGGING
