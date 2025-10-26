#ifndef AMR_INCLUDED_NEIGHBORS
#define AMR_INCLUDED_NEIGHBORS

#include "containers/static_vector.hpp"
#include "ndconcepts.hpp"
#include "ndutils.hpp"
#include <cstddef>
#include <ranges>
#include <type_traits>
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
        static constexpr auto num_neighbors() -> decltype(s_nd_fanout)
        {
            return s_nd_fanout / s_1d_fanout;
        }

        static_assert(num_neighbors() > 0);

        using container_t = std::array<patch_index_t, num_neighbors()>;
        container_t ids;
    };

    using type = std::variant<none, same, finer, coarser>;
    type data  = none{};
};

// TODO: This should be provided by the patch index
template <std::signed_integral auto Dim>
class direction
{
public:
    using index_t   = decltype(Dim);
    using size_type = index_t;

private:
    static constexpr auto s_dimension         = Dim;
    static constexpr auto s_neighbors_per_dim = size_type{ 2 };
    static constexpr auto s_elements          = s_neighbors_per_dim * s_dimension;
    static constexpr auto s_collection = []<auto const... Is>(std::index_sequence<Is...>)
    {
        return std::array<index_t, s_elements>{ index_t{ Is }... };
    }(std::make_index_sequence<std::size_t{ s_elements }>{});

public:
    using vector_t = containers::static_vector<index_t, s_dimension>;

private:
    explicit constexpr direction(index_t const linear_index) noexcept
        : idx_{ linear_index }
    {
    }

public:
    [[nodiscard]]
    static constexpr auto first() noexcept -> direction
    {
        return direction(s_collection.front());
    }

    [[nodiscard]]
    static constexpr auto sentinel() noexcept -> direction
    {
        return direction(s_elements);
    }

    // TODO: This implicitly assumes 2 neighbors per dimension. Can we use the
    // dimension offset?
    [[nodiscard]]
    static constexpr auto is_negative(direction const& d) noexcept -> bool
    {
        return d.index() % s_neighbors_per_dim == 0;
    }

    [[nodiscard]]
    static constexpr auto is_positive(direction const& d) noexcept -> bool
    {
        return !is_negative(d);
    }

    [[nodiscard]]
    static constexpr auto opposite(direction const& d) noexcept -> direction
    {
        return direction(d.idx_ + (is_negative(d) ? index_t{ 1 } : index_t{ -1 }));
    }

    [[nodiscard]]
    static constexpr auto dimension_offset(direction const& d) noexcept -> index_t
    {
        return d.index() % s_neighbors_per_dim;
    }

    [[nodiscard]]
    static constexpr auto unit_vector(direction const& d) noexcept -> index_t
    {
        vector_t ret{};
        ret[d.dimension()] = is_negative(d) ? index_t{ -1 } : index_t{ 1 };
        return ret;
    }

    constexpr auto advance() noexcept -> void
    {
        ++idx_;
    }

    [[nodiscard]]
    constexpr auto index() const noexcept -> index_t
    {
        return idx_;
    }

    [[nodiscard]]
    constexpr auto dimension() const noexcept -> index_t
    {
        return index() / s_neighbors_per_dim;
    }

    [[nodiscard]]
    explicit constexpr operator bool() const noexcept
    {
        return idx_ >= index_t{} && idx_ < s_elements;
    }

    [[nodiscard]]
    constexpr auto operator<=>(direction const&) const noexcept = default;

private:
    index_t idx_;
};

template <concepts::PatchIndex Patch_Index, concepts::PatchLayout Patch_Layout>
class neighbor_utils
{
public:
    using patch_index_t       = Patch_Index;
    using patch_layout_t      = Patch_Layout;
    using neighbor_category_t = neighbor_variant<patch_index_t>;
    using size_type           = patch_layout_t::size_type;
    using index_t             = patch_layout_t::index_t;
    using dimension_t         = patch_layout_t::dimension_t;

private:
    static constexpr auto s_1d_fanout = patch_index_t::fanout();
    static constexpr auto s_nd_fanout = patch_index_t::nd_fanout();
    static constexpr auto s_dimension = patch_index_t::dimension();

public:
    // TODO: This should be provided by the patch index
    using direction_t = direction<std::make_signed_t<index_t>{
        s_dimension }>; // typename patch_index_t::direction_t;

public:
    template <typename T>
    using patch_neighboring_t = std::array<T, 2 * Patch_Index::dimension()>;
    using patch_neighbors_t   = patch_neighboring_t<neighbor_category_t>;
    using child_expansion_t   = containers::utils::types::tensor::
        hypercube_t<patch_index_t, index_t{ s_1d_fanout }, index_t{ s_dimension }>;

private:
    static constexpr auto s_neighbor_relation_maps = []
    {
        using patch_neighbor_relation_t = patch_neighboring_t<NeighborRelation>;
        std::array<patch_neighbor_relation_t, s_nd_fanout> neighbor_relation_maps{};

        for (index_t flat = 0; flat != index_t{ s_nd_fanout }; ++flat)
        {
            auto& relation_array = neighbor_relation_maps[flat];
            // compute multi-index from flat index
            auto const& coords = child_expansion_t::layout_t::multi_index(flat);
            for (auto d = direction_t::first(); d != direction_t::sentinel(); d.advance())
            {
                relation_array[d.index()] =
                    direction_t::is_negative(d)
                        ? ((coords[d.dimension()] == 0) ? NeighborRelation::ParentNeighbor
                                                        : NeighborRelation::Sibling)
                        : ((coords[d.dimension()] == s_1d_fanout - 1)
                               ? NeighborRelation::ParentNeighbor
                               : NeighborRelation::Sibling);
            }
        }
        return neighbor_relation_maps;
    }();

public:
    static constexpr auto compute_fine_boundary_linear_index(
        typename child_expansion_t::multi_index_t const& coords,
        dimension_t                                      dimension
    ) -> index_t
    {
        assert(dimension < coords.rank());

        index_t linear_idx = 0;
        index_t multiplier = 1;

        // Iterate through all dimensions except the specified one
        for (dimension_t d = 0; d < s_dimension; ++d)
        {
            if (d == dimension) // Skip the dimension we want to remove
            {
                continue;
            }
            linear_idx += coords[d] * multiplier;
            multiplier *= s_1d_fanout;
        }
        return linear_idx;
    }

    // TODO: Are there any preconditions to the coords?
    // Seems like some invariants could be broken here
    static auto get_sibling_offset(
        typename child_expansion_t::multi_index_t coords,
        direction_t                               d
    ) noexcept -> typename patch_index_t::offset_t
    {
        const auto dim = d.dimension();
        coords[dim]    = (direction_t::is_positive(d) ? (coords[dim] + 1)
                                                      : coords[dim] + s_1d_fanout - 1) %
                      s_1d_fanout;
        return static_cast<typename patch_index_t::offset_t>(
            child_expansion_t::layout_t::linear_index(coords)
        );
    }

    static constexpr auto compute_boundary_children(direction_t d)
    {
        static constexpr auto num_boundary_children =
            neighbor_category_t::finer::num_neighbors();
        std::array<size_type, num_boundary_children> boundary_children{};

        size_type boundary_idx = 0;
        for (index_t i = 0; i != s_nd_fanout; ++i)
        {
            auto relations = s_neighbor_relation_maps[i];
            if (relations[d.index()] == NeighborRelation::ParentNeighbor)
            {
                boundary_children[boundary_idx++] = i;
            }
        }
        return boundary_children;
    }

    static auto compute_child_neighbors(
        patch_index_t     parent_id,
        patch_neighbors_t parent_neighbor_array,
        const index_t     local_child_id
    ) -> patch_neighbors_t
    {
        auto child_multiindex = child_expansion_t::layout_t::multi_index(local_child_id);
        auto relations        = s_neighbor_relation_maps[local_child_id];
        patch_neighbors_t child_neighbor_array{};

        for (auto d = direction_t::first(); d != direction_t::sentinel(); d.advance())
        {
            auto directional_relation = relations[d.index()];
            if (directional_relation == NeighborRelation::Sibling)
            {
                auto sibling_offset = get_sibling_offset(child_multiindex, d);
                auto sibling_id     = patch_index_t::child_of(parent_id, sibling_offset);
                neighbor_category_t nb;
                nb.data = typename neighbor_category_t::same{ sibling_id };
                child_neighbor_array[d.index()] = nb;
            }
            else
            {
                auto visitor = [&](auto&& neighbor) -> neighbor_category_t
                {
                    using T = std::decay_t<decltype(neighbor)>;

                    if constexpr (std::is_same_v<T, typename neighbor_category_t::none>)
                    {
                        return neighbor_category_t{
                            typename neighbor_category_t::none{}
                        };
                    }
                    else if constexpr (std::is_same_v<
                                           T,
                                           typename neighbor_category_t::same>)
                    {
                        return neighbor_category_t{ typename neighbor_category_t::coarser{
                            neighbor.id } };
                    }
                    else if constexpr (std::is_same_v<
                                           T,
                                           typename neighbor_category_t::finer>)
                    {
                        const auto fine_index = compute_fine_boundary_linear_index(
                            child_multiindex, d.dimension()
                        );
                        auto fine_neighbor_id = neighbor.ids[fine_index];
                        return neighbor_category_t{ typename neighbor_category_t::same{
                            fine_neighbor_id } };
                    }
                    else if constexpr (std::is_same_v<
                                           T,
                                           typename neighbor_category_t::coarser>)
                    {
                        return neighbor_category_t{
                            typename neighbor_category_t::none{}
                        };
                    }
                };

                child_neighbor_array[d.index()] =
                    std::visit(visitor, parent_neighbor_array[d.index()].data);
            }
        }
        return child_neighbor_array;
    }

    static auto compute_parent_neighbors(
        std::array<patch_neighbors_t, s_nd_fanout> child_neighbor_arrays
    ) -> patch_neighbors_t
    {
        patch_neighbors_t parent_neighbor_array{};
        for (auto d = direction_t::first(); d != direction_t::sentinel(); d.advance())
        {
            auto boundary_children = compute_boundary_children(d);

            // Use the first boundary child's neighbor to determine parent's neighbor type
            auto first_boundary_child_neighbor =
                child_neighbor_arrays[boundary_children[0]][d.index()];

            auto visitor = [&](auto&& neighbor) -> neighbor_category_t
            {
                using T = std::decay_t<decltype(neighbor)>;

                if constexpr (std::is_same_v<T, typename neighbor_category_t::none>)
                {
                    return neighbor_category_t{ typename neighbor_category_t::none{} };
                }
                else if constexpr (std::is_same_v<T, typename neighbor_category_t::same>)
                {
                    // Child has same-level neighbor -> parent has finer neighbors
                    typename neighbor_category_t::finer::container_t fine_neighbor_ids{};
                    for (size_type i = 0; i != boundary_children.size(); i++)
                    {
                        auto child_neighbor =
                            child_neighbor_arrays[boundary_children[i]][d.index()];
                        // Extract the same-level neighbor ID from each boundary child
                        std::visit(
                            [&](auto&& child_nb)
                            {
                                using child_t = std::decay_t<decltype(child_nb)>;
                                if constexpr (std::is_same_v<
                                                  child_t,
                                                  typename neighbor_category_t::same>)
                                {
                                    fine_neighbor_ids[i] = child_nb.id;
                                }
                            },
                            child_neighbor.data
                        );
                    }
                    return neighbor_category_t{ typename neighbor_category_t::finer{
                        fine_neighbor_ids } };
                }
                else if constexpr (std::is_same_v<
                                       T,
                                       typename neighbor_category_t::coarser>)
                {
                    // Child has coarser neighbor -> parent has same-level neighbor
                    return neighbor_category_t{ typename neighbor_category_t::same{
                        neighbor.id } };
                }
                else if constexpr (std::is_same_v<T, typename neighbor_category_t::finer>)
                {
                    assert(
                        false &&
                        "Child has finer neighbor during coarsening - unexpected!"
                    );
                    return neighbor_category_t{ typename neighbor_category_t::none{} };
                }
            };

            parent_neighbor_array[d.index()] =
                std::visit(visitor, first_boundary_child_neighbor.data);
        }

        return parent_neighbor_array;
    }

}; // End of neighbor_utils class

} // namespace amr::ndt::neighbors

#endif // AMR_INCLUDED_NEIGHBORS
