#ifndef AMR_INCLUDED_CONTAINER_FUNCTIONS
#define AMR_INCLUDED_CONTAINER_FUNCTIONS

#include "container_concepts.hpp"
#include "utility/error_handling.hpp"
#include <algorithm>
#include <concepts>
#include <functional>
#include <ranges>
#include <type_traits>

namespace amr::containers
{

template <std::floating_point R, concepts::Container C, typename Distance>
    requires std::
        is_invocable_r_v<R, Distance, R, typename C::value_type, typename C::value_type>
    [[nodiscard]]
    constexpr auto
    distance(C const& c1, C const& c2, Distance&& dist_op, R init = R{}) noexcept -> R
{
    assert(std::ranges::size(c1) == std::ranges::size(c2));
    for (auto it1 = std::begin(c1), it2 = std::begin(c2);
         it1 != std::end(c1) && it2 != std::end(c1);
         ++it1, ++it2)
    {
        init = std::invoke_r<R>(std::forward<Distance>(dist_op), init, *it1, *it2);
    }
    return init;
}

/**
 * \brief Returns exactly equals for integral types and nearly equals if else.
 *        Default tolerance used is 1e-4 for non integral types
 */
template <concepts::Container C>
[[nodiscard]]
constexpr auto operator==(C const& c1, C const& c2) noexcept -> bool
{
    if constexpr (std::is_floating_point_v<typename C::value_type>)
    {
        return nearly_equals(c1, c2);
    }
    else
    {
        return exactly_equals(c1, c2);
    }
}

template <concepts::Container C>
[[nodiscard]]
constexpr auto operator!=(C const& c1, C const& c2) noexcept -> bool
{
    return !operator==(c1, c2);
}

template <concepts::Container C>
[[nodiscard]]
constexpr auto nearly_equals(
    C const&                     c1,
    C const&                     c2,
    const typename C::value_type epsilon = 1e-4
) noexcept -> bool
{
    assert(std::ranges::size(c1) == std::ranges::size(c2));
    for (auto it1 = std::begin(c1), it2 = std::begin(c2);
         it1 != std::end(c1) && it2 != std::end(c1);
         ++it1, ++it2)
    {
        if (std::abs(*it1 - *it2) > epsilon) return false;
    }
    return true;
}

template <concepts::Container C>
[[nodiscard]]
constexpr auto exactly_equals(C const& c1, C const& c2) noexcept -> bool
{
    assert(std::ranges::size(c1) == std::ranges::size(c2));
    for (auto it1 = std::begin(c1), it2 = std::begin(c2);
         it1 != std::end(c1) && it2 != std::end(c1);
         ++it1, ++it2)
    {
        if (*it1 != *it2) return false;
    }
    return true;
}

} // namespace amr::containers

#endif // AMR_INCLUDED_CONTAINER_FUNCTIONS
