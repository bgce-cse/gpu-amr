#ifndef AMR_INCLUDED_NDUTILS
#define AMR_INCLUDED_NDUTILS

#include "containers/container_concepts.hpp"
#include "containers/container_utils.hpp"
#include "containers/static_tensor.hpp"
#include "ndconcepts.hpp"
#include "utility/constexpr_functions.hpp"
#include <algorithm>
#include <cassert>
#include <concepts>
#include <type_traits>

namespace amr::ndt::utils
{

[[nodiscard]]
consteval auto
    compute_nd_fanout(std::integral auto dim, std::integral auto fanout) noexcept
    -> std::common_type_t<decltype(dim), decltype(fanout)>
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
        std::remove_const_t<decltype(x)>>
{
    return std::ranges::all_of(r, [x](auto const& e) { return (e) % x == 0; });
}

template <concepts::PatchLayout Layout>
[[nodiscard]]
constexpr auto is_halo_cell(typename Layout::index_t linear_index) noexcept -> bool
{
    using patch_layout_t                    = Layout;
    using layout_t                          = typename patch_layout_t::padded_layout_t;
    using size_type                         = layout_t::size_type;
    static constexpr auto        rank       = layout_t::rank();
    static constexpr auto const& strides    = layout_t::strides();
    static constexpr auto const& sizes      = layout_t::sizes();
    static constexpr size_type   halo_width = patch_layout_t::halo_width();

    assert(linear_index < layout_t::flat_size());

    if (std::is_signed_v<size_type>)
    {
        assert(linear_index >= 0);
    }
    for (auto j = decltype(rank){}; j != rank; ++j)
    {
        const size_type relative_idx = linear_index / strides[j];
        assert(relative_idx < sizes[j]);
        if (relative_idx < halo_width || relative_idx >= sizes[j] - halo_width)
        {
            return true;
        }
        linear_index %= strides[j];
    }
    return false;
}

template <concepts::PatchLayout Patch_Layout, std::integral auto Fanout>
[[nodiscard]]
consteval auto
    fragmentation_patch_maps() noexcept -> containers::utils::types::tensor::hypercube_t<
                                            containers::static_tensor<
                                                typename Patch_Layout::index_t,
                                                typename Patch_Layout::padded_layout_t>,
                                            Fanout,
                                            Patch_Layout::dimension()>
{
    using patch_layout_t         = Patch_Layout;
    static constexpr auto fanout = static_cast<typename patch_layout_t::index_t>(Fanout);

    using layout_t = typename patch_layout_t::padded_layout_t;
    using index_t  = typename patch_layout_t::index_t;
    using tensor_t = typename containers::static_tensor<index_t, layout_t>;
    using patch_shape_t =
        containers::utils::types::tensor::hypercube_t<tensor_t, fanout, tensor_t::rank()>;
    patch_shape_t to{};

    auto                   idx        = typename tensor_t::multi_index_t{};
    static constexpr auto& strides    = layout_t::strides();
    static constexpr auto& data_shape = patch_layout_t::data_layout_t::sizes();
    do
    {
        static constexpr index_t halo_width    = patch_layout_t::halo_width();
        const auto               linear_idx    = layout_t::linear_index(idx);
        const auto               is_halo       = is_halo_cell<patch_layout_t>(linear_idx);
        const auto               offset        = is_halo ? index_t{}
                                                         : std::transform_reduce(
                                          std::cbegin(idx),
                                          std::cend(idx),
                                          std::cbegin(strides),
                                          index_t{},
                                          std::plus{},
                                          [](auto const i, auto const s)
                                          {
                                              assert(i >= halo_width);
                                              return ((i - halo_width) / fanout) * s;
                                          }
                                      );
        auto                     out_patch_idx = typename patch_shape_t::multi_index_t{};
        do
        {
            const auto linear_out_idx      = patch_shape_t::linear_index(out_patch_idx);
            const auto base                = is_halo ? index_t{ -(linear_out_idx + 1) }
                                                     : [&out_patch_idx]()
            {
                index_t ret{};
                for (index_t i = 0; i != layout_t::rank(); ++i)
                {
                    assert(data_shape[i] % fanout == 0);
                    ret += (out_patch_idx[i] * (data_shape[i] / fanout) + halo_width) *
                           strides[i];
                }
                return ret;
            }();
            to[linear_out_idx][linear_idx] = offset + base;
        } while (out_patch_idx.increment());
    } while (idx.increment());
    return to;
}

} // namespace patches

// In ndutils
namespace neighbors
{

enum class NeighborRelation : std::uint8_t
{
    Sibling,
    ParentNeighbor,
};

template <concepts::PatchIndex Patch_Index>
struct neighbor_variant
{
    using patch_index_t               = Patch_Index;
    static constexpr auto s_1d_fanout = patch_index_t::fanout();
    static constexpr auto s_nd_fanout = patch_index_t::nd_fanout();
    static constexpr auto s_dimension = patch_index_t::dimension();

    struct none
    {
    };

    struct same
    {
        patch_index_t id;
    };

    struct coarser
    {
        patch_index_t id;
    };

    struct finer
    {
        static constexpr decltype(s_nd_fanout) s_num_fine = s_nd_fanout / s_1d_fanout;
        using container_t = std::array<patch_index_t, s_num_fine>;
        container_t ids;
    };

    using type = std::variant<none, same, finer, coarser>;
    type data  = none{};
};

template <concepts::PatchIndex Patch_Index>
using neighbor_array_t =
    std::array<neighbor_variant<Patch_Index>, 2 * Patch_Index::dimension()>;

// Helper class for neighbor operations
template <concepts::PatchIndex Patch_Index, concepts::PatchLayout Patch_Layout>
class neighbor_utils
{
public:
    using patch_index_t           = Patch_Index;
    using patch_layout_t          = Patch_Layout;
    using neighbor_variant_t      = neighbor_variant<patch_index_t>;
    using neighbor_array_t        = neighbors::neighbor_array_t<patch_index_t>;
    using patch_index_direction_t = typename patch_index_t::direction_t;
    using size_type               = patch_layout_t::size_type;

    static constexpr auto s_1d_fanout = patch_index_t::fanout();
    static constexpr auto s_nd_fanout = patch_index_t::nd_fanout();
    static constexpr auto s_dimension = patch_index_t::dimension();

    [[nodiscard]]
    static consteval auto compute_neighbor_relation_maps()
    {
        using neighbor_relation_array_t = std::array<NeighborRelation, 2 * s_dimension>;
        std::array<neighbor_relation_array_t, s_nd_fanout> neighbor_relation_maps{};

        for (size_type flat = 0; flat < s_nd_fanout; ++flat)
        {
            auto& relation_array = neighbor_relation_maps[flat];
            // compute multi-index from flat index
            auto coords = compute_child_multi_idx(flat);
            for (size_type d = 0; d < s_dimension; ++d)
            {
                // - direction
                relation_array[2 * d] = (coords[d] == 0)
                                            ? NeighborRelation::ParentNeighbor
                                            : NeighborRelation::Sibling;
                // + direction
                relation_array[2 * d + 1] = (coords[d] == s_1d_fanout - 1)
                                                ? NeighborRelation::ParentNeighbor
                                                : NeighborRelation::Sibling;
            }
        }
        return neighbor_relation_maps;
    }

    static constexpr auto compute_child_multi_idx(size_type linear_idx)
    {
        std::array<size_type, s_dimension> coords{};
        size_type                          remainder = linear_idx;

        // Row-major: last dimension varies fastest
        for (int d = s_dimension - 1; d >= 0; --d)
        {
            coords[d] = remainder % s_1d_fanout;
            remainder /= s_1d_fanout;
        }
        return coords;
    }

    static constexpr auto
        compute_child_linear_idx(std::array<size_type, s_dimension> coords) -> size_type
    {
        size_type linear_idx = 0;
        size_type multiplier = 1;

        // Row-major: last dimension varies fastest
        for (int d = s_dimension - 1; d >= 0; --d)
        {
            linear_idx += coords[d] * multiplier;
            multiplier *= s_1d_fanout;
        }

        return linear_idx;
    }

    static constexpr auto compute_fine_boundary_linear_index(
        std::array<size_type, s_dimension> coords,
        size_type                          dimension
    ) -> size_type
    {
        assert(dimension < coords.size());

        size_type linear_idx = 0;
        size_type multiplier = 1;

        // Iterate through all dimensions except the specified one
        for (size_type d = 0; d < s_dimension; ++d)
        {
            if (d != dimension) // Skip the dimension we want to remove
            {
                linear_idx += coords[d] * multiplier;
                multiplier *= s_1d_fanout;
            }
        }

        return linear_idx;
    }

    static auto
        get_sibling_offset(std::array<size_type, s_dimension> coords, size_type direction)
    {
        size_t dim    = direction / 2;        // Which dimension (0=x, 1=y)
        bool positive = (direction % 2) == 1; // true for +direction, false for -direction

        if (positive)
        {
            coords[dim] = (coords[dim] + 1) % s_1d_fanout;
        }
        else
        {
            coords[dim] = (coords[dim] + s_1d_fanout - 1) % s_1d_fanout;
        }

        return static_cast<patch_index_t::offset_t>(compute_child_linear_idx(coords));
    }

    static constexpr auto compute_boundary_children(size_type direction)
    {
        constexpr size_type num_boundary_children =
            utility::cx_functions::pow(s_1d_fanout, s_dimension - 1);
        std::array<size_type, num_boundary_children> boundary_children{};

        size_type boundary_idx = 0;
        for (size_type i = 0; i < s_nd_fanout; i++)
        {
            auto relations = s_neighbor_relation_maps[i];
            if (relations[direction] == NeighborRelation::ParentNeighbor)
            {
                boundary_children[boundary_idx] = i;
                boundary_idx++;
            }
        }
        return boundary_children;
    }

    static constexpr auto s_neighbor_relation_maps = compute_neighbor_relation_maps();

    static auto compute_child_neighbors(
        patch_index_t    parent_id,
        neighbor_array_t parent_neighbor_array,
        const size_type  local_child_id
    ) -> neighbor_array_t
    {
        auto             child_multiindex = compute_child_multi_idx(local_child_id);
        auto             relations        = s_neighbor_relation_maps[local_child_id];
        neighbor_array_t child_neighbor_array{};

        for (size_t direction = 0; direction < 2 * s_dimension; direction++)
        {
            auto directional_relation = relations[direction];

            if (directional_relation == NeighborRelation::Sibling)
            {
                auto sibling_offset = get_sibling_offset(child_multiindex, direction);
                auto sibling_id     = patch_index_t::child_of(parent_id, sibling_offset);
                neighbor_variant_t nb;
                nb.data = typename neighbor_variant_t::same{ sibling_id };
                child_neighbor_array[direction] = nb;
            }
            else
            {
                auto visitor = [&](auto&& neighbor) -> neighbor_variant_t
                {
                    using T = std::decay_t<decltype(neighbor)>;

                    if constexpr (std::is_same_v<T, typename neighbor_variant_t::none>)
                    {
                        return neighbor_variant_t{ typename neighbor_variant_t::none{} };
                    }
                    else if constexpr (std::is_same_v<
                                           T,
                                           typename neighbor_variant_t::same>)
                    {
                        return neighbor_variant_t{ typename neighbor_variant_t::coarser{
                            neighbor.id } };
                    }
                    else if constexpr (std::is_same_v<
                                           T,
                                           typename neighbor_variant_t::finer>)
                    {
                        auto fine_index = compute_fine_boundary_linear_index(
                            child_multiindex, direction / 2
                        );
                        auto fine_neighbor_id = neighbor.ids[fine_index];
                        return neighbor_variant_t{ typename neighbor_variant_t::same{
                            fine_neighbor_id } };
                    }
                    else if constexpr (std::is_same_v<
                                           T,
                                           typename neighbor_variant_t::coarser>)
                    {
                        return neighbor_variant_t{ typename neighbor_variant_t::none{} };
                    }
                };

                child_neighbor_array[direction] =
                    std::visit(visitor, parent_neighbor_array[direction].data);
            }
        }
        return child_neighbor_array;
    }

    static auto compute_parent_neighbors(
        std::array<neighbor_array_t, s_nd_fanout> child_neighbor_arrays
    ) -> neighbor_array_t
    {
        neighbor_array_t parent_neighbor_array{};

        for (size_type direction = 0; direction < 2 * s_dimension; direction++)
        {
            auto boundary_children = compute_boundary_children(direction);

            // Use the first boundary child's neighbor to determine parent's neighbor type
            auto first_boundary_child_neighbor =
                child_neighbor_arrays[boundary_children[0]][direction];

            auto visitor = [&](auto&& neighbor) -> neighbor_variant_t
            {
                using T = std::decay_t<decltype(neighbor)>;

                if constexpr (std::is_same_v<T, typename neighbor_variant_t::none>)
                {
                    return neighbor_variant_t{ typename neighbor_variant_t::none{} };
                }
                else if constexpr (std::is_same_v<T, typename neighbor_variant_t::same>)
                {
                    // Child has same-level neighbor -> parent has finer neighbors
                    typename neighbor_variant_t::finer::container_t fine_neighbor_ids{};
                    for (size_type i = 0; i < boundary_children.size(); i++)
                    {
                        auto child_neighbor =
                            child_neighbor_arrays[boundary_children[i]][direction];
                        // Extract the same-level neighbor ID from each boundary child
                        std::visit(
                            [&](auto&& child_nb)
                            {
                                using ChildT = std::decay_t<decltype(child_nb)>;
                                if constexpr (std::is_same_v<
                                                  ChildT,
                                                  typename neighbor_variant_t::same>)
                                {
                                    fine_neighbor_ids[i] = child_nb.id;
                                }
                            },
                            child_neighbor.data
                        );
                    }
                    return neighbor_variant_t{ typename neighbor_variant_t::finer{
                        fine_neighbor_ids } };
                }
                else if constexpr (std::
                                       is_same_v<T, typename neighbor_variant_t::coarser>)
                {
                    // Child has coarser neighbor -> parent has same-level neighbor
                    return neighbor_variant_t{ typename neighbor_variant_t::same{
                        neighbor.id } };
                }
                else if constexpr (std::is_same_v<T, typename neighbor_variant_t::finer>)
                {
                    assert(
                        false &&
                        "Child has finer neighbor during coarsening - unexpected!"
                    );
                    return neighbor_variant_t{ typename neighbor_variant_t::none{} };
                }
            };

            parent_neighbor_array[direction] =
                std::visit(visitor, first_boundary_child_neighbor.data);
        }

        return parent_neighbor_array;
    }

}; // End of neighbor_utils class

} // namespace neighbors
} // namespace amr::ndt::utils
#endif // AMR_INCLUDED_NDUTILS
