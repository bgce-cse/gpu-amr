#ifndef AMR_INCLUDED_INTERGRID_OPERATOR
#define AMR_INCLUDED_INTERGRID_OPERATOR

#include "ndconcepts.hpp"
#include <array>
#include <concepts>
#include <iostream>

namespace amr::ndt::intergrid_operator
{

template <concepts::PatchLayout Patch_Layout>
struct linear_interpolator
{
    using patch_layout_t = Patch_Layout;
    using index_t        = typename patch_layout_t::index_t;
    using rank_t         = typename patch_layout_t::rank_t;

    [[gnu::always_inline]]
    static constexpr auto interpolation(
        auto&          to,
        index_t const& to_idx,
        auto const&    from,
        index_t const  from_idx
    ) noexcept -> void
    {
        to[to_idx] = from[from_idx];
    }

    template <std::integral auto N>
    static constexpr auto interpolation(
        auto&                         to,
        std::array<index_t, N> const& to_idxs,
        auto const&                   from,
        index_t const                 from_idx
    ) noexcept -> void
    {
        for (auto const to_idx : to_idxs)
        {
            interpolation(to, to_idx, from, from_idx);
        }
    }

    template <std::integral auto N>
    static constexpr auto restriction(
        auto&                         to,
        index_t const                 to_idx,
        auto const&                   from,
        std::array<index_t, N> const& from_idxs
    ) noexcept -> void
    {
        using value_type = typename std::remove_cvref_t<decltype(from)>::value_type;
        value_type sum{};
        for (auto const idx : from_idxs)
        {
            sum += from[idx];
        }
        to[to_idx] = sum / static_cast<value_type>(N);
    }
};

} // namespace amr::ndt::intergrid_operator

#endif
