#ifndef INCLUDED_STOPWATCH
#define INCLUDED_STOPWATCH

#include <chrono>
#include <iostream>

#ifndef DEBUG_OSTREAM
#define DEBUG_OSTREAM std::cout
#endif

namespace utility::timing
{

class stopwatch
{
    using clock_type = std::chrono::steady_clock;

public:
    stopwatch(const char* func = "Process") :
        function_name_{ func },
        start_{ clock_type::now() }
    {
    }

    stopwatch(stopwatch const&)                    = delete;
    stopwatch(stopwatch&&)                         = delete;
    auto operator=(stopwatch const&) -> stopwatch& = delete;
    auto operator=(stopwatch&&) -> stopwatch&      = delete;

    ~stopwatch()
    {
        const auto duration = clock_type::now() - start_;
        DEBUG_OSTREAM << std::fixed << std::setprecision(4);
        DEBUG_OSTREAM
            << function_name_ << " took "
            << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count()
            << "ms\t ("
            << std::chrono::duration_cast<std::chrono::microseconds>(duration).count()
            << "us)\t ("
            << std::chrono::duration_cast<std::chrono::seconds>(duration).count()
            << " s)\t ("
            << std::chrono::duration_cast<std::chrono::minutes>(duration).count()
            << " mins)\n"
            << std::defaultfloat;
    }

private:
    const char*                  function_name_{};
    const clock_type::time_point start_{};
};

} // namespace utility::timing

#endif // INCLUDED_STOPWATCH
