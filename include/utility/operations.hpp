#ifndef INCLUDED_UTILITY_OPERATIONS
#define INCLUDED_UTILITY_OPERATIONS

#include "casts.hpp"
#include "utility_concepts.hpp"
#include <cmath>

namespace utility::basic_operations
{

template <typename T>
[[nodiscard]]
constexpr auto is_same_sign(T v1, T v2) noexcept -> bool
{
    return (v1 > 0) == (v2 > 0);
}

template <utility::concepts::Arithmetic T>
[[nodiscard]]
auto log_b(T a, T b) noexcept -> T
{
    return utility::casts::safe_cast<T>(std::log(a) / std::log(b));
}

} // namespace utility::basic_operations

#endif // INCLUDED_UTILITY_OPERATIONS
