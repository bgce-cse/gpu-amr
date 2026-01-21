#ifndef INCLUDED_UTILITY_CONTRACTS
#define INCLUDED_UTILITY_CONTRACTS

#include "utility_concepts.hpp"
#include <cassert>
#include <utility>

namespace utility::contracts
{

template <concepts::Arithmetic T>
[[gnu::always_inline]]
constexpr auto assert_range(T const& i, T const& low, T const& high)
{
    if (std::is_constant_evaluated())
    {
        if constexpr (std::is_signed_v<T>)
        {
            if (i < low) std::unreachable();
        }
        if (i >= high) std::unreachable();
    }
#ifndef NDEBUG
    else
    {
        if constexpr (std::is_signed_v<T>)
        {
            assert(i >= low);
        }
        assert(i < high);
    }
#endif
}

template <concepts::Arithmetic T>
[[gnu::always_inline]]
constexpr auto assert_index(T const& i, T const& size)
{
    assert_range(i, T{}, size);
}

} // namespace utility::contracts

#endif // INCLUDED_UTILITY_CONTRACTS
