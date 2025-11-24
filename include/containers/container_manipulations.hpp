#ifndef AMR_INCLUDED_CONTAINER_MANIPULATIONS
#define AMR_INCLUDED_CONTAINER_MANIPULATIONS

#include "container_concepts.hpp"
#include "loop_control.hpp"
#include <algorithm>
#include <concepts>
#include <functional>
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

// TODO: Compare with and without [[gnu::always_inline, gnu::flatten]]
template <
    concepts::LoopControl Loop_Control,
    std::integral auto    I,
    std::integral... Indices>
[[gnu::always_inline, gnu::flatten]]
constexpr auto shaped_for_impl(auto&& fn, Indices... idxs, auto&&... args) noexcept
    -> void
{
    using loop_t = Loop_Control;
    using rank_t = typename loop_t::rank_t;

    if constexpr (I == loop_t::rank())
    {
        static_assert(std::invocable<decltype(fn), decltype(args)..., decltype(idxs)...>);
        std::invoke(
            std::forward<decltype(fn)>(fn), std::forward<decltype(args)>(args)..., idxs...
        );
    }
    else
    {
        for (auto i = loop_t::start(I); i != loop_t::end(I); i += loop_t::stride(I))
        {
            shaped_for_impl<loop_t, I + rank_t{ 1 }, Indices..., decltype(i)>(
                std::forward<decltype(fn)>(fn),
                idxs...,
                i,
                std::forward<decltype(args)>(args)...
            );
        }
    }
}

} // namespace detail

template <concepts::LoopControl Loop_Control>
[[gnu::always_inline, gnu::flatten]]
constexpr auto shaped_for(auto&& fn, auto&&... args) noexcept -> void
{
    using loop_t = Loop_Control;
    using rank_t = typename loop_t::rank_t;
    detail::shaped_for_impl<Loop_Control, rank_t{}>(
        std::forward<decltype(fn)>(fn), std::forward<decltype(args)>(args)...
    );
}


template <concepts::LoopControl Loop_Control>
[[gnu::always_inline, gnu::flatten]]
constexpr auto for_each(auto&& a, auto&& fn, auto&&... args) noexcept -> void
    requires concepts::StaticContainer<std::remove_cvref_t<decltype(a)>>
{
    static_assert(std::is_same_v<
                  typename std::remove_cvref_t<decltype(a)>::shape_t,
                  typename Loop_Control::shape_t>);
    shaped_for<Loop_Control>(
        std::forward<decltype(fn)>(fn),
        std::forward<decltype(a)>(a),
        std::forward<decltype(args)>(args)...
    );
}

[[gnu::always_inline, gnu::flatten]]
constexpr auto apply(auto&& a, auto&& fn, auto&&... args) noexcept -> void
    requires concepts::StaticContainer<std::remove_cvref_t<decltype(a)>>
{
    using s_t  = typename std::remove_cvref_t<decltype(a)>::shape_t;
    using lc_t = control::loop_control<s_t, 0, s_t::sizes(), 1>;
    for_each<lc_t>(
        std::forward<decltype(a)>(a),
        std::forward<decltype(fn)>(fn),
        std::forward<decltype(args)>(args)...
    );
}

} // namespace amr::containers::manipulators

#endif // AMR_INCLUDED_CONTAINER_MANIPULATIONS
