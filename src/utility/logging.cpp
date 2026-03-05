#include "utility/logging.hpp"
#include "spdlog/async.h"
#include "spdlog/sinks/rotating_file_sink.h"
#include <memory>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace utility::logging
{
auto default_logger::init() -> void
{
    static bool initialized = []
    {
        spdlog::init_thread_pool(8192, 1);

        auto stdout_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        auto general_sink =
            std::make_shared<spdlog::sinks::basic_file_sink_mt>("logs/general.log", true);
        auto rotating_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
            "logs/backtrace.log", 1024 * 1024 * 10, 1
        );
        auto errors_sink =
            std::make_shared<spdlog::sinks::basic_file_sink_mt>("logs/errors.log", true);
        errors_sink->set_level(spdlog::level::warn);

        std::vector<spdlog::sink_ptr> async_sinks{ general_sink,
                                                   rotating_sink,
                                                   errors_sink };

        auto logger = std::make_shared<spdlog::logger>("default_source", stdout_sink);
        logger->set_pattern("[%H:%M:%S.%e] [%^%l%$] %v");
        logger->set_level(spdlog::level::info);
        logger->flush_on(spdlog::level::warn);
        logger->enable_backtrace(32);

        auto terminate_handler = []()
        {
            spdlog::critical("std::terminate called");
            spdlog::dump_backtrace();
            spdlog::shutdown();
            std::exit(EXIT_FAILURE);
        };
        std::set_terminate(terminate_handler);

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
        async_logger->info("  type      : {}", LOGGING_BUILD_TYPE);
        async_logger->info("  compiler  : {} {}", LOGGING_COMPILER_ID, LOGGING_COMPILER_VERSION);
        async_logger->info("  execution policy  : {}", LOGGING_EXECUTION);
        async_logger->info("  git hash  : {}", LOGGING_GIT_HASH);
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

auto default_logger::log(spdlog::level::level_enum sev, std::string_view msg) -> void
{
    init();
    if (sev >= spdlog::level::info) // Reduce load for sync sink (stdio)
        default_source()->log(sev, msg);
    default_async_source()->log(sev, msg);
}

} // namespace utility::logging
