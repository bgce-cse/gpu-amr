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

template <
    concepts::LoopControl Loop_Control,
    std::integral auto    I,
    std::integral... Indices>
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

template <concepts::StaticShape S, auto Start, auto End, auto Stride>
class loop_control
{
public:
    using shape_t = S;
    using index_t = typename S::index_t;
    using rank_t  = typename S::rank_t;

private:
    static constexpr auto s_rank = shape_t::rank();

public:
    [[nodiscard]]
    static constexpr auto rank() noexcept -> rank_t
    {
        return shape_t::rank();
    }

private:
    [[nodiscard]]
    static constexpr auto at_idx(auto const& v, const rank_t idx) noexcept
        -> decltype(auto)
    {
        using v_t = std::remove_cvref_t<decltype(v)>;
        if constexpr (std::is_arithmetic_v<v_t>)
        {
            return v;
        }
        else if constexpr (std::ranges::range<v_t>)
        {
            using size_type = typename v_t::size_type;
            return v[static_cast<size_type>(idx)];
        }
        else
        {
            utility::error_handling::assert_unreachable();
        }
    };

    [[nodiscard]]
    static consteval auto check_param(auto const& param) noexcept -> bool
    {
        using param_t = std::remove_cvref_t<decltype(param)>;
        static_assert(std::is_arithmetic_v<param_t> || std::ranges::range<param_t>);
        if constexpr (std::is_arithmetic_v<param_t>)
        {
            return true;
        }
        else if constexpr (std::ranges::range<param_t>)
        {
            return std::ranges::size(param) == s_rank;
        }
        else
        {
            return false;
        }
    }

    [[nodiscard]]
    static consteval auto is_valid() noexcept -> bool
    {
        static_assert(check_param(Start));
        static_assert(check_param(Stride));
        static_assert(check_param(End));
        for (auto i = decltype(s_rank){}; i != s_rank; ++i)
        {
            if ((at_idx(Start, i) >= index_t{}) && (at_idx(Start, i) <= at_idx(End, i)) &&
                (at_idx(End, i) <= shape_t::size(i)) &&
                ((at_idx(End, i) - at_idx(Start, i)) % at_idx(Stride, i) == index_t{}))
            {
            }
            else
            {
                return false;
            }
        }
        return true;
    }

    static_assert(is_valid());

public:
    [[nodiscard]]
    static constexpr auto start(const rank_t i) noexcept -> index_t
    {
        return at_idx(Start, i);
    }

    [[nodiscard]]
    static constexpr auto end(const rank_t i) noexcept -> index_t
    {
        return at_idx(End, i);
    }

    [[nodiscard]]
    static constexpr auto stride(const rank_t i) noexcept -> index_t
    {
        return at_idx(Stride, i);
    }
};

template <concepts::LoopControl Loop_Control>
constexpr auto for_each(auto&& a, auto&& fn, auto&&... args) noexcept -> void
    requires concepts::StaticContainer<std::remove_cvref_t<decltype(a)>>
{
    static_assert(std::is_same_v<
                  typename std::remove_cvref_t<decltype(a)>::shape_t,
                  typename Loop_Control::shape_t>);
    using s_t  = typename std::remove_cvref_t<decltype(a)>::shape_t;
    using rank_t = typename s_t::rank_t;
    detail::shaped_for_impl<Loop_Control, rank_t{}>(
        std::forward<decltype(fn)>(fn),
        std::forward<decltype(a)>(a),
        std::forward<decltype(args)>(args)...
    );
}

constexpr auto apply(auto&& a, auto&& fn, auto&&... args) noexcept -> void
    requires concepts::StaticContainer<std::remove_cvref_t<decltype(a)>>
{
    using s_t  = typename std::remove_cvref_t<decltype(a)>::shape_t;
    using lc_t = loop_control<s_t, 0, s_t::sizes(), 1>;
    for_each<lc_t>(
        std::forward<decltype(a)>(a),
        std::forward<decltype(fn)>(fn),
        std::forward<decltype(args)>(args)...
    );
}

} // namespace amr::containers::manipulators

#endif // AMR_INCLUDED_CONTAINER_MANIPULATIONS
