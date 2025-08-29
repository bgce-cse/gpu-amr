#pragma once

#include <cassert>
#include <chrono>
#include <cmath>
#include <concepts>
#include <format>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>

namespace utils
{

// Because std::pow is always a bottleneck
template <typename T>
[[nodiscard]]
constexpr auto pow2(T x) noexcept -> T
{
    static_assert(std::is_arithmetic_v<T>);
    return x * x;
}

template <typename T>
struct interval
{
    T min;
    T max;
};

/**
 * @brief Defines a value restricted to [0, 1]
 *
 */
template <std::floating_point F>
class unsigned_normalized
{
  public:
    using value_type = F;
    static constexpr auto s_interval = interval<value_type>{0, 1};

    constexpr explicit unsigned_normalized(value_type const& value) noexcept
        : value_(std::clamp(value, s_interval.min, s_interval.max))
    {
    }

    [[nodiscard]]
    constexpr auto value() const noexcept -> value_type
    {
        return value_;
    }

  private:
    value_type value_;
};

/**
 * @brief Defines a value in an open range [a, b]
 *
 * @tparam T The type
 * @param value The value
 * @param range The range
 */
template <typename T>
class ranged_value
{
  public:
    using value_type = T;
    using range_type = interval<T>;

  public:
    constexpr ranged_value(value_type const& value, range_type const& range)
        : range_{range}
        , value_{clamp_impl(value, range)}
    {
    }

    [[nodiscard]]
    constexpr auto get() const noexcept -> value_type
    {
        return value_;
    }

    constexpr auto set(value_type const& value) noexcept -> void
    {
        value_ = clamp_impl(value, range_);
    }

    [[nodiscard]]
    constexpr auto get_range() const noexcept -> range_type
    {
        return range_;
    }

    [[nodiscard]]
    constexpr auto is_max() const noexcept -> bool
    {
        return equal_impl(value_, range_.max);
    }

    [[nodiscard]]
    constexpr auto is_min() const noexcept -> bool
    {
        return equal_impl(value_, range_.min);
    }

    [[nodiscard]]
    constexpr auto percentage() const noexcept
        -> unsigned_normalized<value_type>
    {
        return unsigned_normalized<value_type>(
            (value_ - range_.min) / (range_.max - range_.min)
        );
    }

  private:
    [[nodiscard]]
    static constexpr auto clamp_impl(
        value_type const& value, range_type const& range
    ) noexcept -> value_type
    {
        return clamp(value, range);
    }

    [[nodiscard]]
    static constexpr auto equal_impl(
        value_type const& a, value_type const& b
    ) noexcept -> value_type
    {
        if constexpr (std::is_floating_point_v<value_type>)
        {
            constexpr value_type epsilon =
                std::numeric_limits<value_type>::epsilon();
            return std::abs(a - b) < epsilon;
        }
        else
        {
            return a == b;
        }
    }

  private:
    range_type range_;
    value_type value_;
};

template <typename T>
[[nodiscard]]
constexpr auto clamp(T const& value, interval<T> const& limits) noexcept -> T
{
    return std::clamp(value, limits.min, limits.max);
}

template <typename T>
[[nodiscard]]
auto in(T const& value, interval<T> const& limits) noexcept -> bool
{
    return (value >= limits.min) && (value <= limits.max);
}

/**
 * @brief Returns a value inside a range defined by the percentage of the range
 * covered
 *
 * @tparam T The range value_type
 * @tparam F A compatible type
 * @param interval The open interval [a, b]
 * @param index The point of the range to interpolate in percentage
 * @return The linearly interpolated value between [a, b]
 */
template <typename T, typename F>
[[nodiscard]]
auto index_interval(
    interval<T> const& interval, unsigned_normalized<F> const& index
) -> std::common_type_t<T, F>
{
    using common_type = std::common_type_t<T, F>;
    const auto idx = static_cast<common_type>(index.value());
    const auto min = static_cast<common_type>(interval.min);
    const auto max = static_cast<common_type>(interval.max);
    return static_cast<common_type>(min + (max - min) * idx);
}

/**
 * @brief Loop progress utility class to show the progress of a running loop. It
 * is externally managed and does not execute the loop itself.
 *
 * @tparam Value_Type The value_type used for internal logic
 * @param begin Start point of the loop
 * @param end End point of the loop
 * @return
 */
template <typename Value_Type>
class loop_progress
{
  public:
    using value_type = Value_Type;
    using clock_t = std::chrono::steady_clock;
    using time_point_t = typename clock_t::time_point;
    using duration_t = std::chrono::duration<double>;
    using size_type = int;
    static constexpr auto s_progress_bar_width = size_type{40};

  public:
    loop_progress(value_type begin, value_type end) noexcept
        : _control_var(begin, {begin, end})
        , _start_time{clock_t::now()}
    {
    }

    /**
     * @brief Print loop info
     *
     * @param endchar Char to end the print statement. Typically '\n' or '\t'.
     */
    auto print_progress(char endchar = '\n') const noexcept -> void
    {
        const auto remaining_time_estimate = extrapolate_remainging_time_impl();
        const auto p = _control_var.percentage();
        const auto completed_units =
            static_cast<size_type>(p.value() * s_progress_bar_width);
        std::cout << "Running: [";
        size_type i = 0;
        for (; i != completed_units; ++i)
            std::cout << '=';
        for (; i != s_progress_bar_width; ++i)
            std::cout << ' ';
        std::cout << std::format(
            "] {:>8.2f} seconds remaining{}",
            std::chrono::duration_cast<duration_t>(remaining_time_estimate)
                .count(),
            endchar
        );
    }

    /**
     * @brief Increment the control variable of the loop
     *
     * @param delta The increment step
     */
    inline auto increment(value_type const& delta)
    {
        _control_var.set(_control_var.get() + delta);
    }

  private:
    [[nodiscard]]
    inline auto extrapolate_remainging_time_impl() const noexcept -> duration_t
    {
        const auto now = clock_t::now();
        const auto p = _control_var.percentage();
        assert(p.value() > 0);
        return (now - _start_time) / p.value() * (value_type{1} - p.value());
    };

  private:
    ranged_value<value_type> _control_var;
    time_point_t _start_time;
};

} // namespace utils
