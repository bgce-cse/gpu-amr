#ifndef INCLUDED_CONSTEXPR_FUNNCTIONS
#define INCLUDED_CONSTEXPR_FUNNCTIONS

#include <concepts>
#include <type_traits>

namespace utility::cx_functions
{

// Do not use this for anything serious
template <typename T>
constexpr auto pow(T base, T exp) noexcept -> T
{
    T result{ 1 };
    for (; exp > 0; --exp)
    {
        result *= base;
    }
    return result;
}

template <std::integral auto Num>
    requires std::is_unsigned_v<decltype(Num)>
[[nodiscard]]
constexpr auto log2() noexcept -> decltype(Num)
{
    constexpr auto is_power_of_two = [] [[nodiscard]]
                                     (auto num) constexpr noexcept
    {
        return num > 0 && (num & (num - 1)) == 0;
    };
    static_assert(is_power_of_two(Num));
    decltype(Num) result = 0;
    decltype(Num) n      = Num;
    while (n >>= 1)
    {
        ++result;
    }
    return result;
}

} // namespace utility::cx_functions

#endif // INCLUDED_CONSTEXPR_FUNNCTIONS
