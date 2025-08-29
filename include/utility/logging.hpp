#ifndef INCLUDED_UTILTIY_LOGGING
#define INCLUDED_UTILTIY_LOGGING

#ifdef USE_BOOST_LOGGING

#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/log/attributes.hpp>
#include <boost/log/attributes/scoped_attribute.hpp>
#include <boost/log/attributes/timer.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sinks.hpp>
#include <boost/log/sinks/sync_frontend.hpp>
#include <boost/log/sinks/text_file_backend.hpp>
#include <boost/log/sinks/text_ostream_backend.hpp>
#include <boost/log/sources/record_ostream.hpp>
#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/support/date_time.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/smart_ptr/make_shared_object.hpp>
#include <boost/smart_ptr/shared_ptr.hpp>
#include <fstream>
#include <iomanip>
#include <ios>
#else
#include <string_view>
#ifndef DEBUG_OSTREAM
#include <iostream>
#define DEBUG_OSTREAM std::cout
#endif
#endif

namespace utility::logging
{

// Define our own severity levels
enum severity_level
{
    trace,
    debug,
    info,
    important_info,
    warning,
    error,
    fatal,
};

#ifdef USE_BOOST_LOGGING
namespace log = boost::log;
namespace src = boost::log::sources;
namespace expr = boost::log::expressions;
namespace sinks = boost::log::sinks;
namespace attrs = boost::log::attributes;

#define LOGGING_UTILITY_SCOPED_LOG_TIMELINE                                    \
    BOOST_LOG_SCOPED_THREAD_ATTR("Timeline", attrs::timer());

#define LOGGING_UTILITY_SCOPED_ADD_TAG(tag)                                    \
    BOOST_LOG_SCOPED_THREAD_ATTR("Tag", attrs::constant<std::string>(tag));

// Enable streaming of severity_level enum
auto operator<<(std::ostream& strm, severity_level level) -> std::ostream&
{
    using namespace std::literals;
    static const std::string_view strings[] = {
        "trace"sv,
        "debug"sv,
        "info"sv,
        "important info"sv,
        "warning"sv,
        "error"sv,
        "fatal"sv
    };
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
    using text_sink = sinks::synchronous_sink<sinks::text_ostream_backend>;
    auto general_sink = boost::make_shared<text_sink>();

    general_sink->locked_backend()->add_stream(
        boost::make_shared<std::ofstream>("logs/general.log")
    );

    general_sink->set_formatter(
        expr::stream << std::setw(5) << std::left << std::dec
                     << std::setfill(' ') << expr::attr<unsigned int>("LineID")
                     << " "
                     << expr::format_date_time<boost::posix_time::ptime>(
                            "TimeStamp", "  %Y-%m-%d %H:%M:%S    "
                        )
                     << severity
                     << expr::if_(expr::has_attr(tag_attr)
                        )[expr::stream << "  [" << tag_attr << "]"]
                     << expr::if_(expr::has_attr(timeline)
                        )[expr::stream << "  [" << timeline << "]"]
                     << "  " << expr::smessage
    );

    auto errors_sink = boost::make_shared<text_sink>();
    errors_sink->locked_backend()->add_stream(
        boost::make_shared<std::ofstream>("logs/errors.log")
    );

    errors_sink->set_formatter(
        expr::stream << std::setw(5) << std::left << std::dec
                     << std::setfill(' ') << expr::attr<unsigned int>("LineID")
                     << " "
                     << expr::format_date_time<boost::posix_time::ptime>(
                            "TimeStamp", "  %Y-%m-%d %H:%M:%S    "
                        )
                     << severity
                     << expr::if_(expr::has_attr(tag_attr)
                        )[expr::stream << "  [" << tag_attr << "]"]
                     << expr::if_(expr::has_attr(timeline)
                        )[expr::stream << "  [" << timeline << "]"]
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
  public:
    inline static auto log(severity_level sev, std::string_view message) -> void
    {
#ifdef USE_BOOST_LOGGING
        if (!initialized)
        {
            init();
            initialized = true;
        }
        static src::severity_logger<severity_level> lg;
        BOOST_LOG_SEV(lg, sev) << message;
#else
        DEBUG_OSTREAM << "Log <" << severity_name(sev) << "> " << message
                      << '\n';
#endif
    }

#ifdef USE_BOOST_LOGGING
  private:
    inline static bool initialized = false;
#else

    inline static auto severity_name(severity_level sev
    ) noexcept -> std::string_view
    {
        switch (sev)
        {
        case utility::logging::severity_level::info:
            return "info";
        case utility::logging::severity_level::debug:
            return "debug";
        case utility::logging::severity_level::error:
            return "error";
        case utility::logging::severity_level::fatal:
            return "fatal";
        case utility::logging::severity_level::trace:
            return "trace";
        case utility::logging::severity_level::warning:
            return "warning";
        case utility::logging::severity_level::important_info:
            return "important info";
        default:
            return "UNKNOWN";
        }
    };
#endif
};

} // namespace utility::logging

#endif // INCLUDED_UTILTIY_LOGGING
