#ifndef AMR_INCLUDED_CONTAINER_MANIPULATIONS
#define AMR_INCLUDED_CONTAINER_MANIPULATIONS

#include "container_concepts.hpp"
#include "utility/error_handling.hpp"
#include <algorithm>
#include <concepts>
#include <functional>
#include <ranges>
#include <type_traits>

namespace amr::containers
{

template <concepts::Container C>
constexpr auto fill(C& c, typename C::value_type const& v) noexcept -> void
{
    std::ranges::fill(c, v);
}

template <concepts::Container C, typename Fn, typename... Args>
    requires std::invocable<Fn, Args...> &&
             std::
                 convertible_to<std::invoke_result_t<Fn, Args...>, typename C::value_type>
constexpr auto fill(C& c, Fn&& fn, Args&&... args) noexcept(
    std::is_nothrow_invocable_r_v<typename C::value_type, Fn, Args...>
) -> void
{
    for (auto it = std::begin(c); it != std::end(c); ++it)
    {
        *it = std::invoke(fn, std::forward<Args>(args)...);
    }
}

template <concepts::Container C, typename Fn>
    requires std::is_invocable_r_v<typename C::value_type, Fn, typename C::value_type>
constexpr void transform(C& c, Fn&& fn)
{
    std::transform(begin(c), end(c), begin(c), std::forward<Fn>(fn));
}

} // namespace amr::containers

#endif // AMR_INCLUDED_CONTAINER_MANIPULATIONS
