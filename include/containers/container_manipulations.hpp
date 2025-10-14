#ifndef AMR_INCLUDED_CONTAINER_MANIPULATIONS
#define AMR_INCLUDED_CONTAINER_MANIPULATIONS

#include "container_concepts.hpp"
#include "utility/error_handling.hpp"
#include <algorithm>
#include <concepts>
#include <functional>
#include <ranges>
#include <type_traits>

namespace amr::containers::manipulators
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

namespace detail
{

template <std::integral auto I, typename... Indices>
constexpr auto for_each_impl(auto&& a, auto&& fn, Indices&&... idxs) noexcept -> void
    requires concepts::StaticMDArray<std::remove_cvref_t<decltype(a)>>
{
    using a_t    = std::remove_cvref_t<decltype(a)>;
    using rank_t = typename a_t::rank_t;
    if constexpr (I == a_t::rank())
    {
        std::invoke(
            std::forward<decltype(fn)>(fn),
            std::forward<decltype(a)>(a),
            std::forward<decltype(idxs)>(idxs)...
        );
    }
    else
    {
        for (typename a_t::index_t i = 0; i != a_t::size(I); ++i)
        {
            for_each_impl<I + rank_t{ 1 }>(
                std::forward<decltype(a)>(a),
                std::forward<decltype(fn)>(fn),
                std::forward<decltype(idxs)>(idxs)...,
                i
            );
        }
    }
}

} // namespace detail

constexpr auto apply(auto&& a, auto&& fn) noexcept -> void
    requires concepts::StaticMDArray<std::remove_cvref_t<decltype(a)>>
{
    using a_t    = std::remove_cvref_t<decltype(a)>;
    using rank_t = typename a_t::rank_t;
    detail::for_each_impl<rank_t{}>(
        std::forward<decltype(a)>(a), std::forward<decltype(fn)>(fn)
    );
}

} // namespace amr::containers::manipulators

#endif // AMR_INCLUDED_CONTAINER_MANIPULATIONS
