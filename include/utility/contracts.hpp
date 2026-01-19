#ifndef INCLUDED_UTILITY_CONTRACTS
#define INCLUDED_UTILITY_CONTRACTS

#include "utility_concepts.hpp"
#include <cassert>
#include <utility>

namespace utility::contracts
{

template <concepts::Arithmetic T>
constexpr auto assert_index(T const& i, T const& size)
{
    if (std::is_constant_evaluated())
    {
        if constexpr (std::is_signed_v<T>)
        {
            if (i < T{}) std::unreachable();
        }
        if (i >= size) std::unreachable();
    }
#ifndef NDEBUG
    else
    {
        if constexpr (std::is_signed_v<T>)
        {
            assert(i >= T{});
        }
        assert(i < size);
    }
#endif
}

} // namespace utility::contracts

#endif // INCLUDED_UTILITY_CONTRACTS
