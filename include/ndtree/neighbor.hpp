#ifndef AMR_INCLUDED_NEIGHBORS
#define AMR_INCLUDED_NEIGHBORS

#include "ndconcepts.hpp"
#include "ndutils.hpp"
#include <cstddef>
#include <variant>

namespace amr::ndt::neighbors
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
        // TODO: Maybe store information about relative position?
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

template <concepts::PatchIndex Patch_Index, concepts::PatchLayout Patch_Layout>
class neighbor_utils
{
public:
    using patch_index_t           = Patch_Index;
    using patch_layout_t          = Patch_Layout;
    using neighbor_variant_t      = neighbor_variant<patch_index_t>;
    using patch_index_direction_t = typename patch_index_t::direction_t;
    using size_type               = patch_layout_t::size_type;
    using index_t                 = patch_layout_t::index_t;
    using dimension_t             = patch_layout_t::dimension_t;

private:
    static constexpr auto s_1d_fanout         = patch_index_t::fanout();
    static constexpr auto s_nd_fanout         = patch_index_t::nd_fanout();
    static constexpr auto s_dimension         = patch_index_t::dimension();
    static constexpr auto s_neighbors_per_dim = dimension_t{ 2 };

public:
    template <typename T>
    using patch_neighboring_t = std::array<T, 2 * Patch_Index::dimension()>;
    using patch_neighbors_t   = patch_neighboring_t<neighbor_variant_t>;
    using child_expansion_t   = containers::utils::types::tensor::
        hypercube_t<patch_index_t, index_t{ s_1d_fanout }, index_t{ s_dimension }>;

    static constexpr auto dim_dir_negative(dimension_t const d) noexcept -> dimension_t
    {
        return d * s_neighbors_per_dim;
    }

    static constexpr auto dim_dir_positive(dimension_t const d) noexcept -> dimension_t
    {
        return dim_dir_negative(d) + 1;
    }

    [[nodiscard]]
    static consteval auto compute_neighbor_relation_maps()
    {
        using patch_neighbor_relation_t = patch_neighboring_t<NeighborRelation>;
        std::array<patch_neighbor_relation_t, s_nd_fanout> neighbor_relation_maps{};

        for (index_t flat = 0; flat != index_t{ s_nd_fanout }; ++flat)
        {
            auto& relation_array = neighbor_relation_maps[flat];
            // compute multi-index from flat index
            auto const& coords = child_expansion_t::layout_t::multi_index(flat);
            for (dimension_t d = 0; d != s_dimension; ++d)
            {
                // - direction
                relation_array[dim_dir_negative(d)] =
                    (coords[d] == 0) ? NeighborRelation::ParentNeighbor
                                     : NeighborRelation::Sibling;
                // + direction
                relation_array[dim_dir_positive(d)] =
                    (coords[d] == s_1d_fanout - 1) ? NeighborRelation::ParentNeighbor
                                                   : NeighborRelation::Sibling;
            }
        }
        return neighbor_relation_maps;
    }

    static constexpr auto compute_fine_boundary_linear_index(
        typename child_expansion_t::multi_index_t const& coords,
        dimension_t                                      dimension
    ) -> size_type
    {
        assert(dimension < coords.rank());

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

    static auto get_sibling_offset(
        typename child_expansion_t::multi_index_t coords,
        size_type                                 direction
    ) noexcept -> typename patch_index_t::offset_t
    {
        // TODO: This is terrible
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
        return static_cast<typename patch_index_t::offset_t>(
            child_expansion_t::layout_t::linear_index(coords)
        );
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

    // TODO: This guy got lost. Refactor
    static constexpr auto s_neighbor_relation_maps = compute_neighbor_relation_maps();

    static auto compute_child_neighbors(
        patch_index_t     parent_id,
        patch_neighbors_t parent_neighbor_array,
        const index_t     local_child_id
    ) -> patch_neighbors_t
    {
        auto child_multiindex = child_expansion_t::layout_t::multi_index(local_child_id);
        auto relations        = s_neighbor_relation_maps[local_child_id];
        patch_neighbors_t child_neighbor_array{};

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
        std::array<patch_neighbors_t, s_nd_fanout> child_neighbor_arrays
    ) -> patch_neighbors_t
    {
        patch_neighbors_t parent_neighbor_array{};

        for (size_type direction = 0; direction != s_neighbors_per_dim * s_dimension;
             direction++)
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
                    for (size_type i = 0; i != boundary_children.size(); i++)
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

} // namespace amr::ndt::neighbors

#endif // AMR_INCLUDED_NEIGHBORS
