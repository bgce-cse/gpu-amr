#ifndef INCLUDED_UTILITY_GENERICS
#define INCLUDED_UTILITY_GENERICS

#include <algorithm>
#include <type_traits>

namespace utility::generics
{

template <typename T>
struct interval
{
    T min;
    T max;
};

template <std::floating_point F>
class unsigned_normalized
{
  public:
    static constexpr auto s_interval = interval<F>{0, 1};

    constexpr explicit unsigned_normalized(F value) noexcept
        : value_(std::clamp(value, s_interval.min, s_interval.max))
    {
    }

    [[nodiscard]]
    constexpr auto value() const noexcept -> F
    {
        return value_;
    }

    [[nodiscard]]
    constexpr auto is_max() const noexcept -> bool
    {
        return value_ >= s_interval.max;
    }

    [[nodiscard]]
    constexpr auto is_min() const noexcept -> bool
    {
        return value_ <= s_interval.min;
    }

    [[nodiscard]]
    static constexpr auto max() noexcept -> unsigned_normalized
    {
        return unsigned_normalized(s_interval.max);
    }

    [[nodiscard]]
    static constexpr auto min() noexcept -> unsigned_normalized
    {
        return unsigned_normalized(s_interval.min);
    }

  private:
    F value_;
};

template <typename T>
class ranged_value
{
  public:
    using value_type = T;
    using range_type = interval<T>;

  public:
    constexpr ranged_value(value_type value, range_type range)
        : range_{range}
        , value_{clamp_impl(value, range)}
    {
    }

    [[nodiscard]]
    constexpr auto get() const noexcept -> value_type
    {
        return value_;
    }

    constexpr auto set(value_type value) noexcept -> void
    {
        value_ = clamp_impl(value, range_);
    }

    [[nodiscard]]
    constexpr auto get_range() const noexcept -> range_type
    {
        return range_;
    }

  private:
    [[nodiscard]]
    static constexpr auto clamp_impl(
        value_type value, range_type range
    ) noexcept -> value_type
    {
        return clamp(value, range);
    }

  private:
    range_type range_;
    value_type value_;
};

template <typename T>
[[nodiscard]]
constexpr auto clamp(T value, interval<T> limits) noexcept -> T
{
    return std::clamp(value, limits.min, limits.max);
}

template <typename T>
[[nodiscard]]
constexpr auto in(T value, interval<T> limits) noexcept -> bool
{
    return (value >= limits.min) && (value <= limits.max);
}

template <typename T, typename F>
[[nodiscard]]
constexpr auto index_interval(
    interval<T> const& interval, unsigned_normalized<F> index
) -> std::common_type_t<T, F>
{
    using common_type = std::common_type_t<T, F>;
    const auto idx = static_cast<common_type>(index.value());
    const auto min = static_cast<common_type>(interval.min);
    const auto max = static_cast<common_type>(interval.max);
    return min + (max - min) * idx;
}

} // namespace utility::generics

#endif // INCLUDED_UTILITY_GENERICS
