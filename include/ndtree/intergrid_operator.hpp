#ifndef AMR_INCLUDED_INTERGRID_OPERATOR
#define AMR_INCLUDED_INTERGRID_OPERATOR

#include "ndconcepts.hpp"
#include <array>
#include <concepts>
#include <iostream>

namespace amr::ndt::intergrid_operator
{

/**
 * @brief Linear intergrid operator.
 * Implements averaging for restrinction and copying for interpolation.
 * Interpolation and Restriction APIs are symmetric and provide access to patch
 * internals for flexibility.
 */
template <concepts::PatchLayout Patch_Layout>
struct linear_interpolator
{
    using patch_layout_t = Patch_Layout;
    using index_t        = typename patch_layout_t::index_t;

    /**
     * @brief Single-element interpolation operator.
     * Implements intergrid mapping from from[from_idx] to to[to_idx].
     * This api allows to query neighboring values in order to implement higher
     * order schemes if necessary.
     * Used in halo exchnage since only one element is accessed at a time.
     *
     * @param to Destination patch
     * @param to_idx Linear index of the destination cell in the destination
     * patch
     * @param maybe_unused Linear child offset [0, s_nd_fanout) of the target
     * cell within. This index can come from the linear index of the hypercube
     * expansion type.
     * @param from Source patch
     * @param from_idx Linear index of the source cell in the source patch
     */
    [[gnu::always_inline]]
    static constexpr auto interpolation(
        auto&                           to,
        index_t const&                  to_idx,
        [[maybe_unused]] index_t const& child_offset,
        auto const&                     from,
        index_t const                   from_idx
    ) noexcept -> void
    {
        to[to_idx] = from[from_idx];
    }

    /**
     * @brief Block interpolation operator.
     * Used in tree patch interpolation during reconstruction since the opeation works at
     * a patch level.
     *
     * @param to Destination patch
     * @param to_idxs Set of the linear index of the destination cells in the destination
     * patch. These indices correspond to the cells in the hypercube expansion
     * of a coarse cell into a finer hypercube.
     * @param from Source patch
     * @param from_idx Linear index of the source cell in the source patch
     */
    template <std::integral auto N>
    static constexpr auto interpolation(
        auto&                         to,
        std::array<index_t, N> const& to_idxs,
        auto const&                   from,
        index_t const                 from_idx
    ) noexcept -> void
    {
        for (index_t i{}; i != index_t{ N }; ++i)
        {
            interpolation(to, to_idxs[i], i, from, from_idx);
        }
    }

    /**
     * @brief Restriction operator.
     * Used in both halo exchange and in patch restriction during
     * reconstruction.
     *
     * @param to Destination patch
     * @param to_idx Linear index of the destination cell in the destination
     * patch
     * @param from Source patch
     * @param from_idxs Linear indices of the source cells in the source patch. These
     * indices correspond to the cells in the hypercube expansion of a coarse cell into a
     * finer hypercube.
     */
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
