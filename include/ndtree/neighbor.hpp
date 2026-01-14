#ifndef AMR_INCLUDED_NEIGHBORS
#define AMR_INCLUDED_NEIGHBORS

#include "containers/container_utils.hpp"
#include "containers/static_vector.hpp"
#include "ndconcepts.hpp"
#include "utility/logging.hpp"
#include "../solver/boundary.hpp"
#include <cstddef>
#include <type_traits>
#include <variant>

namespace amr::ndt::neighbors
{

enum class NeighborRelation : std::uint8_t
{
    Sibling,
    ParentNeighbor,
};

template <auto Fanout_1D, auto Fanout_ND, typename Identifier>
struct neighbor_variant
{
    static_assert(std::is_same_v<decltype(Fanout_1D), decltype(Fanout_ND)>);
    using identifier_t = Identifier;
    using fanout_t     = decltype(Fanout_1D);
    static_assert(std::is_same_v<decltype(Fanout_1D), decltype(Fanout_ND)>);

    static constexpr fanout_t s_1d_fanout = Fanout_1D;
    static constexpr fanout_t s_nd_fanout = Fanout_ND;

    struct none
    {
    };

    struct same
    {
        identifier_t id;
    };

    struct coarser
    {
        identifier_t id;
        fanout_t     dim_offset;
    };

    struct finer
    {
        static constexpr auto num_neighbors() -> decltype(s_nd_fanout)
        {
            return s_nd_fanout / s_1d_fanout;
        }

        static_assert(num_neighbors() > 0);

        template <typename T>
        using container_type_t = std::array<T, num_neighbors()>;
        using container_t      = container_type_t<identifier_t>;
        container_t ids;
    };

    using type = std::variant<none, same, finer, coarser>;
    type data  = none{};
};

// TODO: This should be provided by the patch index
template <std::integral auto Dim>
struct direction
{
public:
    using index_t   = decltype(Dim);
    using size_type = index_t;

private:
    static constexpr size_type s_rank              = Dim;
    static constexpr auto      s_neighbors_per_dim = size_type{ 2 };
    static constexpr size_type s_elements          = s_neighbors_per_dim * s_rank;
    static constexpr auto s_collection = []<auto const... Is>(std::index_sequence<Is...>)
    {
        return std::array<index_t, s_elements>{ index_t{ Is }... };
    }(std::make_index_sequence<std::size_t{ s_elements }>{});

public:
    using signed_index_t = std::make_signed_t<index_t>;
    using vector_t       = containers::static_vector<signed_index_t, s_rank>;

private:
    explicit constexpr direction(index_t const linear_index) noexcept
        : idx_{ linear_index }
    {
    }

public:
    [[nodiscard]]
    static constexpr auto rank() noexcept -> decltype(s_rank)
    {
        return s_rank;
    }

    [[nodiscard]]
    static constexpr auto elements() noexcept -> decltype(s_elements)
    {
        return s_elements;
    }

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
        if (!is_negative(d)) [[assume(d.idx_ > index_t{})]];
        return direction(is_negative(d) ? d.idx_ + index_t{ 1 } : d.idx_ - index_t{ 1 });
    }

    [[nodiscard]]
    static constexpr auto advance(direction d) noexcept -> direction
    {
        d.advance();
        return d;
    }

    [[nodiscard]]
    static constexpr auto dimension_offset(direction const& d) noexcept -> index_t
    {
        return d.index() % s_neighbors_per_dim;
    }

    [[nodiscard]]
    static constexpr auto unit_vector(direction const& d) noexcept -> vector_t
    {
        vector_t ret{};
        ret[d.dimension()] = is_negative(d) ? signed_index_t{ -1 } : signed_index_t{ 1 };
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

public:
    index_t idx_;
};

template <concepts::PatchIndex Patch_Index, concepts::PatchLayout Patch_Layout>
class neighbor_utils
{
public:
    using patch_index_t  = Patch_Index;
    using patch_layout_t = Patch_Layout;
    

private:
    static constexpr auto s_1d_fanout = patch_index_t::fanout();
    static constexpr auto s_nd_fanout = patch_index_t::nd_fanout();
    static constexpr auto s_rank      = patch_index_t::rank();

public:
    template <typename Id_Type>
    using neighbor_variant_base_t = neighbor_variant<s_1d_fanout, s_nd_fanout, Id_Type>;
    using neighbor_variant_t      = neighbor_variant_base_t<patch_index_t>;
    using size_type               = patch_layout_t::size_type;
    using index_t                 = patch_layout_t::index_t;
    using rank_t                  = patch_layout_t::rank_t;

public:
    // TODO: This should be provided by the patch index
    using direction_t =
        direction<index_t{ s_rank }>; // typename patch_index_t::direction_t;
    
    using bc_type_array_t = std::array<amr::ndt::solver::bc_type, direction_t::elements()>;

public:
    template <typename T>
    using patch_neighboring_t = std::array<T, direction_t::elements()>;
    using patch_neighbors_t   = patch_neighboring_t<neighbor_variant_t>;
    using child_expansion_t   = containers::utils::types::tensor::
        hypercube_t<patch_index_t, index_t{ s_1d_fanout }, index_t{ s_rank }>;

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
        rank_t                                           rank
    ) -> index_t
    {
        assert(rank < coords.rank());

        index_t linear_idx = 0;
        index_t multiplier = 1;

        // Iterate through all dimensions except the specified one
        for (rank_t d = 0; d != s_rank; ++d)
        {
            if (d == rank)
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
            neighbor_variant_t::finer::num_neighbors();
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

    [[nodiscard]]
    static auto compute_child_neighbors(
        patch_index_t const&     parent_id,
        patch_neighbors_t const& parent_neighbor_array,
        index_t const&           local_child_id,
        bc_type_array_t          bc_types
    ) -> patch_neighbors_t
    {
        DEFAULT_SOURCE_LOG_TRACE(
            std::string("Computing child neighbors for parent ") + parent_id.repr() + 
            ", child index " + std::to_string(local_child_id)
        );
        
        const auto child_multiindex =
            child_expansion_t::layout_t::multi_index(local_child_id);
        const auto        relations = s_neighbor_relation_maps[local_child_id];
        patch_neighbors_t child_neighbor_array{};

        for (auto d = direction_t::first(); d != direction_t::sentinel(); d.advance())
        {
            const auto directional_relation = relations[d.index()];
            
            DEFAULT_SOURCE_LOG_TRACE(
                std::string("  Direction ") + std::to_string(d.index()) + ": " +
                (directional_relation == NeighborRelation::Sibling ? "Sibling" : "ParentNeighbor")
            );
            
            if (directional_relation == NeighborRelation::Sibling ||
                (parent_id == patch_index_t::root() &&
                 bc_types[d.index()] == amr::ndt::solver::bc_type::Periodic))
            {
                const auto sibling_offset = get_sibling_offset(child_multiindex, d);
                const auto sibling_id =
                    patch_index_t::child_of(parent_id, sibling_offset);
                DEFAULT_SOURCE_LOG_TRACE(
                    std::string("    Setting sibling neighbor: ") + sibling_id.repr()
                );
                neighbor_variant_t nb;
                nb.data = typename neighbor_variant_t::same{ sibling_id };
                child_neighbor_array[d.index()] = nb;
            }
            else
            {
                auto visitor = [&](auto const& neighbor) -> neighbor_variant_t
                {
                    using T = std::decay_t<decltype(neighbor)>;

                    if constexpr (std::is_same_v<T, typename neighbor_variant_t::none>)
                    {
                        DEFAULT_SOURCE_LOG_TRACE("    Parent has none -> child has none");
                        return neighbor_variant_t{ typename neighbor_variant_t::none{} };
                    }
                    else if constexpr (std::is_same_v<
                                           T,
                                           typename neighbor_variant_t::same>)
                    {
                        DEFAULT_SOURCE_LOG_TRACE("    Parent has same -> child has coarser");
                        return neighbor_variant_t{
                            typename neighbor_variant_t::coarser{
                                                                 neighbor.id,
                                                                 static_cast<typename neighbor_variant_t::fanout_t>(
                                    compute_fine_boundary_linear_index(
                                        child_multiindex, d.dimension()
                                    )
                                ) }
                        };
                    }
                    else if constexpr (std::is_same_v<
                                           T,
                                           typename neighbor_variant_t::finer>)
                    {
                        DEFAULT_SOURCE_LOG_TRACE("    Parent has finer -> child has same");
                        const auto fine_index = compute_fine_boundary_linear_index(
                            child_multiindex, d.dimension()
                        );
                        auto fine_neighbor_id = neighbor.ids[fine_index];
                        return neighbor_variant_t{ typename neighbor_variant_t::same{
                            fine_neighbor_id } };
                    }
                    else if constexpr (std::is_same_v<
                                           T,
                                           typename neighbor_variant_t::coarser>)
                    {
                        DEFAULT_SOURCE_LOG_TRACE("    Parent has coarser -> child has none");
                        return neighbor_variant_t{ typename neighbor_variant_t::none{} };
                    }
                };

                child_neighbor_array[d.index()] =
                    std::visit(visitor, parent_neighbor_array[d.index()].data);
            }
        }
        
        DEFAULT_SOURCE_LOG_TRACE("Child neighbor computation complete");
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
                            child_neighbor_arrays[boundary_children[i]][d.index()];
                        // Extract the same-level neighbor ID from each boundary child
                        std::visit(
                            [&](auto&& child_nb)
                            {
                                using child_t = std::decay_t<decltype(child_nb)>;
                                if constexpr (std::is_same_v<
                                                  child_t,
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

            parent_neighbor_array[d.index()] =
                std::visit(visitor, first_boundary_child_neighbor.data);
        }

        return parent_neighbor_array;
    }

}; // End of neighbor_utils class

} // namespace amr::ndt::neighbors

#endif // AMR_INCLUDED_NEIGHBORS
