#ifndef INCLUDED_UTILITY_LOGGING
#define INCLUDED_UTILITY_LOGGING

#include "macro_definitions.hpp"
#include <format>
#include <spdlog/common.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <string_view>

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
        utility::logging::default_source::log(                   \
            spdlog::level::trace, fmt __VA_OPT__(, ) __VA_ARGS__ \
        )
#else
#    define DEFAULT_SOURCE_LOG_TRACE(...) DEFAULT_SOURCE_LOG_IGNORE(__VA_ARGS__)
#endif

#if DEFAULT_SOURCE_LOG_LEVEL <= DEFAULT_SOURCE_LOG_LEVEL_DEBUG
#    define DEFAULT_SOURCE_LOG_DEBUG(fmt, ...)                   \
        utility::logging::default_source::log(                   \
            spdlog::level::debug, fmt __VA_OPT__(, ) __VA_ARGS__ \
        )
#else
#    define DEFAULT_SOURCE_LOG_DEBUG(...) DEFAULT_SOURCE_LOG_IGNORE(__VA_ARGS__)
#endif

#if DEFAULT_SOURCE_LOG_LEVEL <= DEFAULT_SOURCE_LOG_LEVEL_INFO
#    define DEFAULT_SOURCE_LOG_INFO(fmt, ...)                   \
        utility::logging::default_source::log(                  \
            spdlog::level::info, fmt __VA_OPT__(, ) __VA_ARGS__ \
        )
#else
#    define DEFAULT_SOURCE_LOG_INFO(...) DEFAULT_SOURCE_LOG_IGNORE(__VA_ARGS__)
#endif

#if DEFAULT_SOURCE_LOG_LEVEL <= DEFAULT_SOURCE_LOG_LEVEL_WARNING
#    define DEFAULT_SOURCE_LOG_WARNING(fmt, ...)                \
        utility::logging::default_source::log(                  \
            spdlog::level::warn, fmt __VA_OPT__(, ) __VA_ARGS__ \
        )
#else
#    define DEFAULT_SOURCE_LOG_WARNING(...) DEFAULT_SOURCE_LOG_IGNORE(__VA_ARGS__)
#endif

#if DEFAULT_SOURCE_LOG_LEVEL <= DEFAULT_SOURCE_LOG_LEVEL_ERROR
#    define DEFAULT_SOURCE_LOG_ERROR(fmt, ...)                 \
        utility::logging::default_source::log(                 \
            spdlog::level::err, fmt __VA_OPT__(, ) __VA_ARGS__ \
        )
#else
#    define DEFAULT_SOURCE_LOG_ERROR(...) DEFAULT_SOURCE_LOG_IGNORE(__VA_ARGS__)
#endif

#if DEFAULT_SOURCE_LOG_LEVEL <= DEFAULT_SOURCE_LOG_LEVEL_FATAL
#    define DEFAULT_SOURCE_LOG_FATAL(fmt, ...)                      \
        utility::logging::default_source::log(                      \
            spdlog::level::critical, fmt __VA_OPT__(, ) __VA_ARGS__ \
        )
#else
#    define DEFAULT_SOURCE_LOG_FATAL(...) DEFAULT_SOURCE_LOG_IGNORE(__VA_ARGS__)
#endif

namespace utility::logging
{

class default_source
{
public:
    static auto init() -> void
    {
        static bool initialized = []
        {
            auto console = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
            auto general = std::make_shared<spdlog::sinks::basic_file_sink_mt>(
                "logs/general.log", true
            );
            auto errors = std::make_shared<spdlog::sinks::basic_file_sink_mt>(
                "logs/errors.log", true
            );

            errors->set_level(spdlog::level::warn);

            std::vector<spdlog::sink_ptr> sinks{ console, general, errors };

            auto logger =
                std::make_shared<spdlog::logger>("amr", sinks.begin(), sinks.end());

            logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
            logger->set_level(spdlog::level::trace);
            logger->flush_on(spdlog::level::warn);

            spdlog::register_logger(logger);
            spdlog::set_default_logger(logger);

            logger->info("Build:");
            logger->info("  type      : {}", AMR_BUILD_TYPE);
            logger->info("  compiler  : {} {}", AMR_COMPILER_ID, AMR_COMPILER_VERSION);
            logger->info("  git hash  : {}", AMR_GIT_HASH);
            logger->info("Runtime configuration:");
            logger->info(
                "  log level : {}", spdlog::level::to_string_view(logger->level())
            );
            logger->info("  threads   : {}", std::thread::hardware_concurrency());
            logger->info("  pid       : {}", ::getpid());

            return true;
        }();

        (void)initialized;
    }

    static auto log(spdlog::level::level_enum sev, auto&& fmt, auto&&... args) -> void
    {
        init();
        if constexpr (requires {
                          std::format(
                              std::forward<decltye(fmt)>(fmt),
                              std::forward<decltype(args)>(args)...
                          );
                      })
        {
            spdlog::log(
                sev,
                std::format(
                    std::forward<decltype(fmt)>(fmt),
                    std::forward<decltype(args)>(args)...
                )
            );
        }
        else
        {
            spdlog::log(
                sev, std::vformat(std::string_view(fmt), std::make_format_args(args...))
            );
        }
    }
};

} // namespace utility::logging

#endif // INCLUDED_UTILITY_LOGGING
