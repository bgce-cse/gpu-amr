#ifndef INCLUDED_CONSTEXPR_FUNNCTIONS
#define INCLUDED_CONSTEXPR_FUNNCTIONS

#include "utility_concepts.hpp"
#include <concepts>
#include <type_traits>

namespace utility::cx_functions
{

constexpr auto pow(concepts::Arithmetic auto base, concepts::Arithmetic auto exp) noexcept
    -> decltype(base)
{
    using ret_t = decltype(base);
    ret_t result{ 1 };
    for (; exp > 0; --exp)
    {
        result *= base;
    }
    return result;
}

[[nodiscard]]
consteval auto bits_for(std::unsigned_integral auto n) noexcept -> decltype(n)
{
    using T  = decltype(n);
    T result = 1;
    while (pow(T{ 2 }, result) < n)
    {
        ++result;
    }
    return result;
}

} // namespace utility::cx_functions

#endif // INCLUDED_CONSTEXPR_FUNNCTIONS
