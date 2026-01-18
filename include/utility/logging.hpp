#ifndef INCLUDED_UTILTIY_LOGGING
#define INCLUDED_UTILTIY_LOGGING

#include "macro_definitions.hpp"

#ifdef USE_BOOST_LOGGING

#    include <boost/date_time/posix_time/posix_time_types.hpp>
#    include <boost/log/attributes.hpp>
#    include <boost/log/attributes/scoped_attribute.hpp>
#    include <boost/log/attributes/timer.hpp>
#    include <boost/log/expressions.hpp>
#    include <boost/log/sinks.hpp>
#    include <boost/log/sinks/sync_frontend.hpp>
#    include <boost/log/sinks/text_file_backend.hpp>
#    include <boost/log/sinks/text_ostream_backend.hpp>
#    include <boost/log/sources/record_ostream.hpp>
#    include <boost/log/sources/severity_logger.hpp>
#    include <boost/log/support/date_time.hpp>
#    include <boost/log/trivial.hpp>
#    include <boost/log/utility/setup/common_attributes.hpp>
#    include <boost/log/utility/setup/file.hpp>
#    include <boost/smart_ptr/make_shared_object.hpp>
#    include <boost/smart_ptr/shared_ptr.hpp>
#    include <fstream>
#    include <iomanip>
#    include <ios>
#    include <mutex>
#else
#    include <string_view>
#    ifndef DEBUG_OSTREAM
#        include <iostream>
#        define DEBUG_OSTREAM std::cout
#    endif
#endif

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
#ifndef AMR_DEFAULR_LOG_LEVEL_ERROR
#    define AMR_DEFAULR_LOG_LEVEL_ERROR 5
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
#define DEFAULT_SOURCE_LOG_LEVEL_ERROR   AMR_DEFAULR_LOG_LEVEL_ERROR
#define DEFAULT_SOURCE_LOG_LEVEL_FATAL   AMR_DEFAULT_LOG_LEVEL_FATAL
#define DEFAULT_SOURCE_LOG_LEVEL_OFF     AMR_DEFAULT_LOG_LEVEL_OFF

#ifdef AMR_LOG_LEVEL
#    define DEFAULT_SOURCE_LOG_LEVEL \
        UTILITY_CONCATENATE_MACRO(DEFAULT_SOURCE_LOG_LEVEL_, AMR_LOG_LEVEL)
#else
#    define DEFAULT_SOURCE_LOG_LEVEL DEFAULT_SOURCE_LOG_LEVEL_INFO
#endif

#define DEFAULT_SOURCE_LOG_IGNORE(msg) ((void)0)

#if DEFAULT_SOURCE_LOG_LEVEL <= DEFAULT_SOURCE_LOG_LEVEL_TRACE
#    define DEFAULT_SOURCE_LOG_TRACE(msg)                \
        utility::logging::default_source::log(           \
            utility::logging::severity_level::trace, msg \
        )
#else
#    define DEFAULT_SOURCE_LOG_TRACE(msg) DEFAULT_SOURCE_LOG_IGNORE(msg)
#endif

#if DEFAULT_SOURCE_LOG_LEVEL <= DEFAULT_SOURCE_LOG_LEVEL_DEBUG
#    define DEFAULT_SOURCE_LOG_DEBUG(msg)                \
        utility::logging::default_source::log(           \
            utility::logging::severity_level::debug, msg \
        )
#else
#    define DEFAULT_SOURCE_LOG_DEBUG(msg) DEFAULT_SOURCE_LOG_IGNORE(msg)
#endif

#if DEFAULT_SOURCE_LOG_LEVEL <= DEFAULT_SOURCE_LOG_LEVEL_INFO
#    define DEFAULT_SOURCE_LOG_INFO(msg) \
        utility::logging::default_source::log(utility::logging::severity_level::info, msg)
#else
#    define DEFAULT_SOURCE_LOG_INFO(msg) DEFAULT_SOURCE_LOG_IGNORE(msg)
#endif

#if DEFAULT_SOURCE_LOG_LEVEL <= DEFAULT_SOURCE_LOG_LEVEL_WARNING
#    define DEFAULT_SOURCE_LOG_WARNING(msg)                \
        utility::logging::default_source::log(             \
            utility::logging::severity_level::warning, msg \
        )
#else
#    define DEFAULT_SOURCE_LOG_WARNING(msg) DEFAULT_SOURCE_LOG_IGNORE(msg)
#endif

#if DEFAULT_SOURCE_LOG_LEVEL <= DEFAULT_SOURCE_LOG_LEVEL_ERROR
#    define DEFAULT_SOURCE_LOG_ERROR(msg)                \
        utility::logging::default_source::log(           \
            utility::logging::severity_level::error, msg \
        )
#else
#    define DEFAULT_SOURCE_LOG_ERROR(msg) DEFAULT_SOURCE_LOG_IGNORE(msg)
#endif

#if DEFAULT_SOURCE_LOG_LEVEL <= DEFAULT_SOURCE_LOG_LEVEL_FATAL
#    define DEFAULT_SOURCE_LOG_FATAL(msg)                \
        utility::logging::default_source::log(           \
            utility::logging::severity_level::fatal, msg \
        )
#else
#    define DEFAULT_SOURCE_LOG_FATAL(msg) DEFAULT_SOURCE_LOG_IGNORE(msg)
#endif

namespace utility::logging
{
// Define our own severity levels
enum severity_level
{
    trace,
    debug,
    info,
    warning,
    error,
    fatal,
};

#ifdef USE_BOOST_LOGGING
namespace log   = boost::log;
namespace src   = boost::log::sources;
namespace expr  = boost::log::expressions;
namespace sinks = boost::log::sinks;
namespace attrs = boost::log::attributes;

#    define LOGGING_UTILITY_SCOPED_LOG_TIMELINE \
        BOOST_LOG_SCOPED_THREAD_ATTR("Timeline", attrs::timer());

#    define LOGGING_UTILITY_SCOPED_ADD_TAG(tag) \
        BOOST_LOG_SCOPED_THREAD_ATTR("Tag", attrs::constant<std::string>(tag));

// Enable streaming of severity_level enum
auto operator<<(std::ostream& strm, severity_level level) -> std::ostream&
{
    using namespace std::literals;
    static const std::string_view strings[] = { "trace"sv,   "debug"sv, "info"sv,
                                                "warning"sv, "error"sv, "fatal"sv };
    if (static_cast<std::size_t>(level) < sizeof(strings) / sizeof(*strings))
    {
        const auto s = strings[level];
        strm << "<" << s << std::setw(static_cast<int>(9uz - s.size()))
             << std::setfill(' ') << ">";
    }
    else
    {
        strm << "<" << static_cast<int>(level) << ">";
    }
    return strm;
}

// Define attribute keywords
BOOST_LOG_ATTRIBUTE_KEYWORD(line_id, "LineID", unsigned int)
BOOST_LOG_ATTRIBUTE_KEYWORD(severity, "Severity", severity_level)
BOOST_LOG_ATTRIBUTE_KEYWORD(tag_attr, "Tag", std::string_view)
BOOST_LOG_ATTRIBUTE_KEYWORD(scope, "Scope", attrs::named_scope::value_type)
BOOST_LOG_ATTRIBUTE_KEYWORD(timeline, "Timeline", attrs::timer::value_type)

//[ example_tutorial_formatters_stream_date_time
auto init() -> void
{
    using text_sink   = sinks::synchronous_sink<sinks::text_ostream_backend>;
    auto general_sink = boost::make_shared<text_sink>();

    general_sink->locked_backend()->add_stream(
        boost::make_shared<std::ofstream>("logs/general.log")
    );

    general_sink->set_formatter(
        expr::stream
        << std::setw(5) << std::left << std::dec << std::setfill(' ')
        << expr::attr<unsigned int>("LineID") << " "
        << expr::format_date_time<boost::posix_time::ptime>(
               "TimeStamp", "  %Y-%m-%d %H:%M:%S    "
           )
        << severity
        << expr::if_(expr::has_attr(tag_attr))[expr::stream << "  [" << tag_attr << "]"]
        << expr::if_(expr::has_attr(timeline))[expr::stream << "  [" << timeline << "]"]
        << "  " << expr::smessage
    );

    auto errors_sink = boost::make_shared<text_sink>();
    errors_sink->locked_backend()->add_stream(
        boost::make_shared<std::ofstream>("logs/errors.log")
    );

    errors_sink->set_formatter(
        expr::stream
        << std::setw(5) << std::left << std::dec << std::setfill(' ')
        << expr::attr<unsigned int>("LineID") << " "
        << expr::format_date_time<boost::posix_time::ptime>(
               "TimeStamp", "  %Y-%m-%d %H:%M:%S    "
           )
        << severity
        << expr::if_(expr::has_attr(tag_attr))[expr::stream << "  [" << tag_attr << "]"]
        << expr::if_(expr::has_attr(timeline))[expr::stream << "  [" << timeline << "]"]
        << "  " << expr::smessage
    );
    errors_sink->set_filter(severity >= severity_level::warning);
    general_sink->locked_backend()->auto_flush(true);
    errors_sink->locked_backend()->auto_flush(true);

    log::core::get()->add_sink(general_sink);
    log::core::get()->add_sink(errors_sink);

    log::add_common_attributes();
}
#endif

class default_source
{
    inline static constexpr std::string_view s_info_repr    = "info";
    inline static constexpr std::string_view s_debug_repr   = "debug";
    inline static constexpr std::string_view s_error_repr   = "error";
    inline static constexpr std::string_view s_fatal_repr   = "fatal";
    inline static constexpr std::string_view s_trace_repr   = "trace";
    inline static constexpr std::string_view s_warning_repr = "warning";
    inline static constexpr std::string_view s_unknown_repr = "UNKNOWN";

public:
    inline static auto log(severity_level sev, auto&& message) -> void
    {
#ifdef USE_BOOST_LOGGING
        std::cal_once(s_initialized, init);
        static src::severity_logger<severity_level> lg;
        BOOST_LOG_SEV(lg, sev) << std::forward<decltype(message)>(message);
#else
        DEBUG_OSTREAM << "Log <" << severity_name(sev) << "> "
                      << std::forward<decltype(message)>(message) << '\n';
#endif
    }

#ifdef USE_BOOST_LOGGING
private:
    inline static bool std::once_flag s_initialized;
#else

    inline static auto severity_name(severity_level sev) noexcept -> std::string_view
    {
        switch (sev)
        {
            case utility::logging::severity_level::info: return s_info_repr;
            case utility::logging::severity_level::debug: return s_debug_repr;
            case utility::logging::severity_level::error: return s_error_repr;
            case utility::logging::severity_level::fatal: return s_fatal_repr;
            case utility::logging::severity_level::trace: return s_trace_repr;
            case utility::logging::severity_level::warning: return s_warning_repr;
            default: return s_unknown_repr;
        }
    };
#endif
};

} // namespace utility::logging

#endif // INCLUDED_UTILTIY_LOGGING
