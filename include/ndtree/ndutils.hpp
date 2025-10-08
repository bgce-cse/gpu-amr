#ifndef AMR_INCLUDED_NDUTILS
#define AMR_INCLUDED_NDUTILS

#include "containers/container_utils.hpp"
#include "containers/static_layout.hpp"
#include "containers/static_tensor.hpp"
#include "ndconcepts.hpp"
#include "utility/constexpr_functions.hpp"
#include <algorithm>
#include <concepts>
#include <type_traits>







namespace amr::ndt::utils
{

    enum class NeighborRelation : uint8_t {
    Sibling,
    ParentNeighbor,
};


template <size_t Fanout, size_t Dim, size_t ND_Fanout>
[[nodiscard]]
consteval auto compute_neighbor_relation_maps()
{

    using neighbor_relation_array_t = std::array<NeighborRelation, 2 * Dim>;
    std::array<neighbor_relation_array_t, ND_Fanout> neighbor_relation_maps{};

    for (std::size_t flat = 0; flat < ND_Fanout; ++flat)
    {
        neighbor_relation_array_t relation_array{};

        // compute multi-index from flat index
        std::array<std::size_t, Dim> coords{};
        std::size_t remainder = flat;
        for (std::size_t d = 0; d < Dim; ++d)
        {
            coords[d] = remainder % Fanout;
            remainder /= Fanout;
        }

        for (std::size_t d = 0; d < Dim; ++d)
        {
            // - direction
            relation_array[2*d] = (coords[d] == 0)
                ? NeighborRelation::ParentNeighbor
                : NeighborRelation::Sibling;

            // + direction
            relation_array[2*d + 1] = (coords[d] == Fanout - 1)
                ? NeighborRelation::ParentNeighbor
                : NeighborRelation::Sibling;
        }

        neighbor_relation_maps[flat] = relation_array;
    }

    return neighbor_relation_maps;
}



[[nodiscard]]
consteval auto subdivisions(
    std::unsigned_integral auto dim,
    std::unsigned_integral auto fanout
) noexcept -> std::common_type_t<decltype(dim), decltype(fanout)>
{
    using return_t = std::common_type_t<decltype(dim), decltype(fanout)>;
    return utility::cx_functions::pow(return_t{ dim }, return_t{ fanout });
}

namespace patches
{

[[nodiscard]]
consteval auto multiples_of(
    std::ranges::input_range auto const& r,
    std::integral auto const             x
) noexcept -> bool
    requires std::convertible_to<
        std::ranges::range_value_t<std::remove_cvref_t<decltype(r)>>,
        decltype(x)>
{
    return std::ranges::all_of(r, [x](auto const& e) { return (e) % x == 0; });
}

template <
    std::integral      Index_Type,
    std::integral auto Fanout,
    std::integral auto H,
    std::integral auto N,
    std::integral auto... Ns>
[[nodiscard]]
consteval auto fragmentation_patch_maps(containers::static_layout<H, N, Ns...>) noexcept
    -> containers::utils::types::tensor::hypercube_t<
        containers::static_tensor<Index_Type, 0, N, Ns...>,  // Maps use halo=0
        0,      // ← Add halo parameter for hypercube
        Fanout,
        containers::static_tensor<Index_Type, 0, N, Ns...>::s_rank>
{
    using index_t  = Index_Type;
    using tensor_t = containers::static_tensor<index_t, 0, N, Ns...>;  // Maps don't need halos
    using patch_shape_t =
        containers::utils::types::tensor::hypercube_t<tensor_t, 0, Fanout, tensor_t::s_rank>;
    //                                                        ↑ halo=0 for maps
    patch_shape_t ret{};

    auto idx           = typename tensor_t::multi_index_t{};
    auto sized_strides = decltype(tensor_t::s_strides){};
    std::transform(
        std::cbegin(tensor_t::s_strides),
        std::cend(tensor_t::s_strides),
        std::cbegin(tensor_t::s_sizes),
        std::begin(sized_strides),
        std::multiplies{}
    );
    do
    {
        const auto offset = std::transform_reduce(
            std::cbegin(idx),
            std::cend(idx),
            std::cbegin(tensor_t::s_strides),
            index_t{},
            std::plus{},
            [](index_t const i, index_t const s) { return (i / Fanout) * s; }
        );
        auto out_patch_idx = typename patch_shape_t::multi_index_t{};
        do
        {
            const auto base = std::transform_reduce(
                                  std::cbegin(out_patch_idx),
                                  std::cend(out_patch_idx),
                                  std::cbegin(sized_strides),
                                  index_t{}
                              ) /
                              Fanout;
            ret[out_patch_idx][idx] = offset + base;
        } while (out_patch_idx.increment());
    } while (idx.increment());
    return ret;
}

} // namespace patches

} // namespace amr::ndt::utils

#endif // AMR_INCLUDED_NDUTILS
