#ifndef INCLUDED_CONSTEXPR_FUNNCTIONS
#define INCLUDED_CONSTEXPR_FUNNCTIONS

#include "utility_concepts.hpp"
#include <concepts>
#include <type_traits>

namespace utility::cx_functions
{

// Do not use this for anything serious
template <concepts::Arithmetic T>
constexpr auto pow(T base, T exp) noexcept -> T
{
    T result{1};
    for (; exp > 0; --exp)
    {
        result *= base;
    }
    return result;
}

[[nodiscard]]
consteval auto bits_for(std::unsigned_integral auto const n
) noexcept -> decltype(n)
{
    using T = std::remove_cvref_t<decltype(n)>;
    T result = 1;
    while (pow(T{2}, result) < n)
    {
        ++result;
    }
    return result;
}

} // namespace utility::cx_functions

#endif // INCLUDED_CONSTEXPR_FUNNCTIONS
