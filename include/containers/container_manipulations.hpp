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

template <concepts::LoopControl Loop_Control, std::integral auto I, typename... Indices>
constexpr auto for_each_impl(auto&& a, auto&& fn, Indices&&... idxs) noexcept -> void
    requires concepts::StaticMDArray<std::remove_cvref_t<decltype(a)>>
{
    using a_t    = std::remove_cvref_t<decltype(a)>;
    using loop_t = Loop_Control;
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
        for (auto i = loop_t::start(I); i != loop_t::end(I); i += loop_t::stride(I))
        {
            for_each_impl<loop_t, I + rank_t{ 1 }>(
                std::forward<decltype(a)>(a),
                std::forward<decltype(fn)>(fn),
                std::forward<decltype(idxs)>(idxs)...,
                i
            );
        }
    }
}

} // namespace detail

template <concepts::StaticMDArray A, auto Start, auto End, auto Stride>
class loop_control
{
private:
    static constexpr auto s_rank = A::rank();

public:
    using index_t = typename A::index_t;
    using rank_t  = typename A::rank_t;

    [[nodiscard]]
    static constexpr auto at_idx(auto const& v, rank_t idx) noexcept -> decltype(auto)
    {
        using v_type = std::remove_cvref_t<decltype(v)>;
        if constexpr (std::is_arithmetic_v<v_type>)
        {
            return v;
        }
        else if constexpr (std::ranges::range<v_type>)
        {
            return v[idx];
        }
        else
        {
            utility::error_handling::assert_unreachable();
        }
    };

    static constexpr auto start(const rank_t i) noexcept -> index_t
    {
        return at_idx(Start, i);
    }

    static constexpr auto end(const rank_t i) noexcept -> index_t
    {
        return at_idx(End, i);
    }

    static constexpr auto stride(const rank_t i) noexcept -> index_t
    {
        return at_idx(Stride, i);
    }

    [[nodiscard]]
    static constexpr auto is_valid() noexcept -> bool
    {
        if constexpr (std::is_arithmetic_v<decltype(Start)>)
        {
        }
        else
        {
            static_assert(
                std::ranges::range<decltype(Start)> && std::ranges::size(Start) == s_rank
            );
        }
        if constexpr (std::is_arithmetic_v<decltype(End)>)
        {
        }
        else
        {
            static_assert(
                std::ranges::range<decltype(End)> && std::ranges::size(End) == s_rank
            );
        }
        if constexpr (std::is_arithmetic_v<decltype(Stride)>)
        {
        }
        else
        {
            static_assert(
                std::ranges::range<decltype(Stride)> &&
                std::ranges::size(Stride) == s_rank
            );
        }
        for (auto i = decltype(s_rank){}; i != s_rank; ++i)
        {
            if ((at_idx(Start, i) >= index_t{}) && (at_idx(Start, i) <= at_idx(End, i)) &&
                (at_idx(End, i) <= A::size(i)) &&
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
};

constexpr auto apply(auto&& a, auto&& fn) noexcept -> void
    requires concepts::StaticMDArray<std::remove_cvref_t<decltype(a)>>
{
    using a_t    = std::remove_cvref_t<decltype(a)>;
    using rank_t = typename a_t::rank_t;
    using lc_t   = loop_control<a_t, 0, a_t::sizes(), 1>;
    detail::for_each_impl<lc_t, rank_t{}>(
        std::forward<decltype(a)>(a), std::forward<decltype(fn)>(fn)
    );
}

} // namespace amr::containers::manipulators

#endif // AMR_INCLUDED_CONTAINER_MANIPULATIONS
