#include "utility/logging.hpp"

#ifdef ENABLE_SPDLOG

#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include <memory>
#include <thread>
#include <unistd.h>
#include <vector>

namespace utility::logging
{

static void dump_bt_on_exit()
{
    if (auto* lg = spdlog::default_logger_raw())
    {
        for (auto& sink : lg->sinks())
        {
            sink->set_level(spdlog::level::trace);
        }
        lg->dump_backtrace();
        lg->flush();
    }
}

auto default_logger::init() -> void
{
    static bool initialized = []
    {
        auto stdout_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        stdout_sink->set_level(spdlog::level::info);
        stdout_sink->set_pattern("[%H:%M:%S.%e] [%^%l%$] %v");

        auto general_sink =
            std::make_shared<spdlog::sinks::basic_file_sink_mt>("logs/general.log", true);
        general_sink->set_level(spdlog::level::trace);
        general_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");

        auto errors_sink =
            std::make_shared<spdlog::sinks::basic_file_sink_mt>("logs/errors.log", true);
        errors_sink->set_level(spdlog::level::warn);
        errors_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");

        std::vector<spdlog::sink_ptr> sinks{ stdout_sink, general_sink, errors_sink };

        auto logger =
            std::make_shared<spdlog::logger>("default", sinks.begin(), sinks.end());

        logger->set_level(spdlog::level::trace);
        logger->flush_on(spdlog::level::warn);
        logger->enable_backtrace(32);

        spdlog::register_logger(logger);
        spdlog::set_default_logger(logger);

        logger->info("Build:");
        logger->info("  type      : {}", LOGGING_BUILD_TYPE);
        logger->info(
            "  compiler  : {} {}", LOGGING_COMPILER_ID, LOGGING_COMPILER_VERSION
        );
        logger->info("  execution policy : {}", LOGGING_EXECUTION);
        logger->info("  git hash  : {}", LOGGING_GIT_HASH);
        logger->info("Runtime configuration:");
        logger->info("  log level : {}", spdlog::level::to_string_view(logger->level()));
        logger->info("  threads   : {}", std::thread::hardware_concurrency());
        logger->info("  pid       : {}", ::getpid());

        std::atexit(dump_bt_on_exit);

        return true;
    }();

    (void)initialized;
}

auto default_logger::log(spdlog::level::level_enum sev, std::string_view msg) -> void
{
    init();
    auto* lg = spdlog::default_logger_raw();
    lg->log(sev, msg);
}

} // namespace utility::logging

#endif // ENABLE_SPDLOG
