#ifndef AMR_INCLUDED_NDTREE
#define AMR_INCLUDED_NDTREE

#include "ndconcepts.hpp"
#include "ndhierarchy.hpp"
#include "ndutils.hpp"
#include "utility/compile_time_utility.hpp"
#include "utility/constexpr_functions.hpp"
#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <system_error>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>

#ifndef NDEBUG
#    define AMR_NDTREE_CHECKBOUNDS
#    define AMR_NDTREE_ENABLE_CHECKS
// #    define AMR_NDTREE_CHECK_NEIGHBORS
// #    define AMR_NDTREE_CHECK_BALANCING
#endif




namespace amr::ndt::tree
{

template <
    concepts::DeconstructibleType T,
    concepts::PatchIndex          Patch_Index,
    concepts::StaticLayout         Patch_Layout>
class ndtree
{
public:
    using value_type                  = T;
    using size_type                   = std::size_t;
    using patch_index_t               = Patch_Index;
    using patch_index_directon_t      = typename patch_index_t::direction_t;
    using linear_index_t              = size_type;
    using patch_layout_t              = Patch_Layout;
    static constexpr auto s_nd_fanout = patch_index_t::nd_fanout();
    static constexpr auto s_fanout      = patch_index_t::fanout();
    static constexpr auto s_dimension   = patch_layout_t::s_rank;

    static_assert(s_nd_fanout > 1);
    static_assert(
        utils::patches::multiples_of(patch_layout_t::layout_t::s_logical_sizes, patch_index_t::fanout()),
        "All patch dimensions must be multiples of the fanout"
    );

    template <typename Type>
    using value_t = std::remove_pointer_t<std::remove_cvref_t<Type>>;
    template <typename Type>
    using pointer_t = Type*;
    template <typename Type>
    using const_pointer_t = Type const*;
    template <typename Type>
    using reference_t = Type&;
    template <typename Type>
    using const_reference_t = Type const&;

    template <typename>
    struct deconstructed_buffers_impl;

    template <typename... Ts>
        requires concepts::detail::type_map_tuple_impl<std::tuple<Ts...>>
    struct deconstructed_buffers_impl<std::tuple<Ts...>>
    {
        using type = std::tuple<pointer_t<typename Ts::type>...>;
    };

    using deconstructed_buffers_t =
        typename deconstructed_buffers_impl<typename T::deconstructed_types_map_t>::type;

    template <typename>
    struct deconstructed_types_impl;

    template <typename... Ts>
        requires concepts::detail::type_map_tuple_impl<std::tuple<Ts...>>
    struct deconstructed_types_impl<std::tuple<Ts...>>
    {
        using type = std::tuple<value_t<typename Ts::type>...>;
    };

    using deconstructed_types_t =
        typename deconstructed_types_impl<typename T::deconstructed_types_map_t>::type;

    enum struct RefinementStatus : char
    {
        Stable  = 0,
        Refine  = 1,
        Coarsen = 2,
    };
    using refine_status_t = RefinementStatus;

    using linear_index_map_t         = pointer_t<patch_index_t>;
    using linear_index_array_t       = pointer_t<linear_index_t>;
    using flat_refine_status_array_t = pointer_t<refine_status_t>;
    using index_map_t                = std::unordered_map<patch_index_t, linear_index_t>;
    using index_map_iterator_t       = typename index_map_t::iterator;
    using index_map_const_iterator_t = typename index_map_t::const_iterator;
    


    using static_child_layout_t  =  amr::containers::static_layout<0, s_fanout,s_fanout >; // needs to be done generically but idk how size is fanout ^dimension


    struct NeighborVariant
    {

        struct None { };                      
        struct Same { patch_index_t id; };       
        struct Coarser { patch_index_t id; };    

        static constexpr size_t s_num_fine =
            []{ std::size_t n = 1; for (size_t i = 0; i < s_dimension - 1; ++i) n *= s_fanout; return n; }();

        struct Finer { std::array<patch_index_t, s_num_fine> ids; };

        using type = std::variant<None, Same, Finer, Coarser>;

        type data = None{};
    };

    using neighbor_array_t       = std::array<NeighborVariant,2*s_dimension> ;
    using neighbor_buffer_t           = pointer_t<neighbor_array_t>;

    static constexpr auto s_neighbor_relation_maps = amr::ndt::utils::compute_neighbor_relation_maps<s_fanout,s_dimension, s_nd_fanout>();

    static constexpr auto s_patch_maps = 
        amr::ndt::utils::patches::fragmentation_patch_maps<linear_index_t, patch_index_t::fanout()>(
            typename patch_layout_t::layout_t{}
        );

public:
    ndtree(size_type size) noexcept
        : m_size{}

    {
        std::apply(
            [size](auto&... b)
            {
                ((void)(b = (pointer_t<value_t<decltype(b)>>)
                            std::malloc(size * sizeof(value_t<decltype(b)>))),
                 ...);
            },
            m_data_buffers
        );
        m_linear_index_map =
            (pointer_t<patch_index_t>)std::malloc(size * sizeof(patch_index_t));
        m_reorder_buffer =
            (pointer_t<linear_index_t>)std::malloc(size * sizeof(linear_index_t));
        m_refine_status_buffer =
            (pointer_t<refine_status_t>)std::malloc(size * sizeof(refine_status_t));
        std::iota(m_reorder_buffer, &m_reorder_buffer[size], 0);

        m_neighbors = (neighbor_buffer_t)std::malloc(size * sizeof(NeighborVariant));

        neighbor_array_t root_neighbor_array{};  
        append(patch_index_t::root(), root_neighbor_array);
    }

    ~ndtree() noexcept
    {
        std::free(m_refine_status_buffer);
        std::free(m_reorder_buffer);
        std::free(m_linear_index_map);
        std::free(m_neighbors);
        std::apply([](auto&... b) { ((void)std::free(b), ...); }, m_data_buffers);
    }

public:
    [[nodiscard]]
    auto size() const noexcept -> size_type
    {
        return m_size;
    }

    template <concepts::TypeMap Map_Type>
    [[nodiscard, gnu::always_inline, gnu::flatten]]
    auto get(linear_index_t const idx) noexcept -> reference_t<typename Map_Type::type>
    {
        assert(idx < m_size);
        return std::get<Map_Type::index()>(m_data_buffers)[idx];
    }

    template <concepts::TypeMap Map_Type>
    [[nodiscard, gnu::always_inline, gnu::flatten]]
    auto get(linear_index_t const idx) const noexcept
        -> const_reference_t<typename Map_Type::type>
    {
        assert(idx < m_size);
        return std::get<Map_Type::index()>(m_data_buffers)[idx];
    }


     template <concepts::TypeMap Map_Type>
    [[nodiscard, gnu::always_inline, gnu::flatten]]
    auto get_patch(linear_index_t const patch_idx) noexcept 
        -> patch_layout_t& 
    {
        assert(patch_idx < m_size);
        auto* data_start = &std::get<Map_Type::index()>(m_data_buffers)[patch_idx * patch_layout_t::s_flat_size];
        
        // Return a reference to the data reinterpreted as a tensor
        return *reinterpret_cast<patch_layout_t*>(data_start);
    }

    template <concepts::TypeMap Map_Type>
    [[nodiscard, gnu::always_inline, gnu::flatten]]
    auto get_patch(linear_index_t const patch_idx) const noexcept 
        -> const patch_layout_t& 
    {
        assert(patch_idx < m_size);
        auto* data_start = &std::get<Map_Type::index()>(m_data_buffers)[patch_idx * patch_layout_t::s_flat_size];
        
        return *reinterpret_cast<const patch_layout_t*>(data_start);
    }

    auto fragment(patch_index_t const node_id) -> void
    {
        const auto it = find_index(node_id);
        assert(it.has_value());
        auto const start_to = m_size * patch_layout_t::flat_size() ;
        for (auto i = decltype(s_nd_fanout){}; i != s_nd_fanout; ++i)
        {
            auto child_id = patch_index_t::child_of(node_id, i);
            assert(!find_index(child_id).has_value());
            neighbor_array_t neighbor_array = compute_child_neighbors(node_id, i);
            enforce_symmetric_neighbors(child_id, neighbor_array);
            append(child_id,neighbor_array);
            assert(m_index_map[child_id] == back_idx());
            assert(m_linear_index_map[back_idx()] == child_id);
        }
        for (auto i = decltype(s_nd_fanout){}; i != s_nd_fanout; ++i)
        {
            auto child_id = patch_index_t::child_of(node_id, i);
            enforce_symmetric_neighbors(child_id, m_neighbors[m_index_map[child_id]]);
        }


        const auto from = it.value()->second * patch_layout_t::flat_size();
        interpolate_patch(from, start_to);
        m_index_map.erase(it.value());
#ifdef AMR_NDTREE_ENABLE_CHECKS
        check_index_map();
#endif
    }

    auto recombine(patch_index_t const parent_node_id) -> void
    {
        assert(!find_index(parent_node_id).has_value());

        const auto child_0    = patch_index_t::child_of(parent_node_id, 0);
        const auto child_0_it = find_index(child_0);
        assert(child_0_it.has_value());

        const auto start = child_0_it.value()->second * patch_layout_t::flat_size();
        auto const to = m_size * patch_layout_t::flat_size() ;
        neighbor_array_t empty{};
        append(parent_node_id, empty);
        assert(m_linear_index_map[back_idx()] == parent_node_id);
        restrict_patches(start, to);

        for (auto i = decltype(s_nd_fanout){}; i != s_nd_fanout; ++i)
        {
            const auto child_i    = patch_index_t::child_of(parent_node_id, i);
            auto       child_i_it = find_index(child_i);
            assert(child_i_it.has_value());
            // assert(child_i_it.value()->second == start + i);
            m_index_map.erase(child_i_it.value());
        }
    }

    auto fragment() -> void
    {
        assert(is_sorted());
        for (auto const& node_id : m_to_refine)
        {
            fragment(node_id);
        }
        // print tree for debugging

        sort_buffers();
    }

    auto recombine() -> void
    {
        assert(is_sorted());
        for (const auto& node_id : m_to_coarsen)
        {
            recombine(node_id);
        }
        sort_buffers();
    }


    auto compute_child_neighbors(patch_index_t parent_id, size_t local_child_id) -> neighbor_array_t
    {
        // Get the linear index of the parent to access its neighbors
        auto parent_it = find_index(parent_id);
        assert(parent_it.has_value() && "Parent must exist in the tree");
        linear_index_t parent_linear_idx = parent_it.value()->second;
        auto local_multiindex = static_child_layout_t::multi_index(local_child_id);
        auto relations = s_neighbor_relation_maps[local_child_id];
        neighbor_array_t child_neighbor_array{};
        neighbor_array_t parent_neighbor_array = m_neighbors[parent_linear_idx];

        for (size_t direction = 0; direction < 2 * s_dimension; direction++)
        {
            auto directional_relation = relations[direction];
            
            if (directional_relation == amr::ndt::utils::NeighborRelation::Sibling)
            {
                // Calculate which sibling child this direction points to
               auto  sibling_offset = get_sibling_offset(local_child_id, direction);
                auto sibling_id = patch_index_t::child_of(parent_id, sibling_offset);
                
                // FIXED: Use the same pattern as your working solution
                NeighborVariant nb;
                nb.data = typename NeighborVariant::Same{ sibling_id };
                child_neighbor_array[direction] = nb;
            }
            else // ParentNeighbor case
            {
                NeighborVariant parent_directional_neighbor = parent_neighbor_array[direction];
                
                // Handle different types of parent neighbors
                std::visit([&](auto&& neighbor) {
                    using Neighbor_T = std::decay_t<decltype(neighbor)>;
                    
                    if constexpr (std::is_same_v<Neighbor_T, typename NeighborVariant::None>) {
                        // Parent has no neighbor in this direction (boundary)
                        NeighborVariant nb;
                        nb.data = typename NeighborVariant::None{};
                        child_neighbor_array[direction] = nb;
                    }
                    else if constexpr (std::is_same_v<Neighbor_T, typename NeighborVariant::Same>) {
                        // Parent's neighbor is at same level -> becomes coarser for child
                        NeighborVariant nb;
                        // FIXED: Only access .id for types that have it
                        if constexpr (requires { neighbor.id; }) {
                            nb.data = typename NeighborVariant::Coarser{ neighbor.id };
                        }
                        child_neighbor_array[direction] = nb;
                    }
                    else if constexpr (std::is_same_v<Neighbor_T,typename NeighborVariant::Finer>) {

                            size_t direction_dim = direction / 2;  // 0=x, 1=y for 2D
                            size_t relevant_coord = 0;
                            if constexpr (s_dimension == 2) {
                                relevant_coord = local_multiindex[(direction_dim + 1) % s_dimension];
                            } else{
                                assert(false);
                            }
                            
                            // The fine neighbor at this position becomes the same-level neighbor
                            auto fine_neighbor_id = neighbor.ids[relevant_coord];
                            
                            // Create same-level neighbor relationship
                            NeighborVariant nb;
                            nb.data = typename NeighborVariant::Same{ fine_neighbor_id };
                            child_neighbor_array[direction] = nb;
                        }
                    else if constexpr (std::is_same_v<Neighbor_T,typename NeighborVariant::Coarser>) {
            
                        assert(false && "sth isogin wrong as your neighbro is coarser and u try to refine my friend");
                    }
                    else {
                        assert(false && "Unknown neighbor variant type");
                    }
                }, parent_directional_neighbor.data);
            }
        }
        
        return child_neighbor_array;
    }


    typename patch_index_t::offset_t get_sibling_offset(size_t local_child_id, size_t direction) const
{
    // Convert local_child_id to coordinates
    std::array<typename patch_index_t::offset_t, s_dimension> coords{};
    size_t remainder = local_child_id;
    for (size_t d = 0; d < s_dimension; ++d) {
        coords[d] = remainder % s_fanout;
        remainder /= s_fanout;
    }
    
    // Calculate sibling coordinates based on direction
    size_t dim = direction / 2;  // Which dimension (0=x, 1=y)
    bool positive = (direction % 2) == 1;  // true for +direction, false for -direction
    
    if (positive) {
        coords[dim] = (coords[dim] + 1) % s_fanout;
    } else {
        coords[dim] = (coords[dim] + s_fanout - 1) % s_fanout;
    }
    
    // Convert back to flat index
    typename patch_index_t::offset_t result = 0;
    typename patch_index_t::offset_t multiplier = 1;
    for (size_t d = 0; d < s_dimension; ++d) {
        result += coords[d] * multiplier;
        multiplier *= s_fanout;
    }
    
    return result;
}


// Helper function to get specific fine neighbors for a child
 patch_index_t get_fine_neighbors_for_child(
    const std::array<patch_index_t, NeighborVariant::s_num_fine>& fine_neighbors,
    size_t local_child_id, 
    size_t direction) const
{
    auto relevant_coord = local_child_id / s_fanout *(direction / 2);

    std::array<size_t, s_dimension> coords{};
    size_t remainder = local_child_id;
    for (size_t d = 0; d < s_dimension; ++d) {
        coords[d] = remainder % s_fanout;
        remainder /= s_fanout;
    }
    return fine_neighbors[relevant_coord];

}

 void enforce_symmetric_neighbors(patch_index_t patch_id, neighbor_array_t& neighbor_array){

    for (size_t direction = 0; direction < 2 * s_dimension; direction++) {
        auto& neighbor_variant = neighbor_array[direction]; // retrieve variant
        
        // Calculate opposite direction
        size_t opposite_direction;
        if (direction % 2 == 0) {
            opposite_direction = direction + 1; // -x -> +x, -y -> +y
        } else {
            opposite_direction = direction - 1; // +x -> -x, +y -> -y
        }
        
        // Handle different neighbor types
        std::visit([&](auto&& neighbor) {
            using Neighbor_T = std::decay_t<decltype(neighbor)>;
            
            if constexpr (std::is_same_v<Neighbor_T, typename NeighborVariant::None>) {
                // No neighbor - nothing to enforce
                return;
            }
            else if constexpr (std::is_same_v<Neighbor_T, typename NeighborVariant::Same>) {
                // Single same-level neighbor
                enforce_neighbor_symmetry_single(patch_id, neighbor.id, opposite_direction);
            }
            else if constexpr (std::is_same_v<Neighbor_T, typename NeighborVariant::Coarser>) {
                // Single coarser neighbor - patch_id should appear in the coarser neighbor's Finer list
                enforce_neighbor_symmetry_coarser(patch_id, neighbor.id, opposite_direction);
            }
            else{
                assert(false);
            }
        }, neighbor_variant.data);
    }
}

private:
// Helper function for same-level neighbor symmetry
void enforce_neighbor_symmetry_single(patch_index_t patch_id, patch_index_t neighbor_id, 
                                    size_t opposite_direction) {
    auto neighbor_it = find_index(neighbor_id);
    if (!neighbor_it.has_value()) {
        return; // Neighbor doesn't exist in tree yet
    }
    
    linear_index_t neighbor_linear_idx = neighbor_it.value()->second;
    auto& neighbor_neighbor_array = m_neighbors[neighbor_linear_idx];
    
    // Always overwrite with correct symmetry
    NeighborVariant nb;
    nb.data = typename NeighborVariant::Same{ patch_id };
    neighbor_neighbor_array[opposite_direction] = nb;
}

// Helper function for coarser neighbor symmetry
void enforce_neighbor_symmetry_coarser(patch_index_t patch_id, patch_index_t coarser_neighbor_id,
                                     size_t opposite_direction) {
    auto patch_local_offset = patch_index_t::offset_of(patch_id);
    auto multi_index = static_child_layout_t::multi_index(patch_local_offset);

    auto neighbor_it = find_index(coarser_neighbor_id);
    if (!neighbor_it.has_value()) {
        assert(false); 
    }
    
    linear_index_t neighbor_linear_idx = neighbor_it.value()->second;
    auto& neighbor_neighbor_array = m_neighbors[neighbor_linear_idx];
    
    // FIXED: Fill ALL fine neighbors on the boundary face
    std::array<patch_index_t, NeighborVariant::s_num_fine> finer_ids{};
    
    // Get the parent of patch_id to find all siblings on the boundary
    auto parent_id = patch_index_t::parent_of(patch_id);
    
    // Determine which coordinate is parallel to the boundary (the one that varies)
    size_t direction_dim = opposite_direction / 2;  // 0=x, 1=y for 2D
    size_t parallel_dim = (direction_dim + 1) % s_dimension;  // The perpendicular dimension
    
    // Fill all fine neighbors along the boundary face
    for (size_t i = 0; i < s_fanout; i++) {
        // Create coordinates for each sibling on the boundary
        std::array<typename patch_index_t::offset_t, s_dimension> sibling_coords{};
        
        // Copy the fixed coordinate from our patch
        sibling_coords[direction_dim] = multi_index[direction_dim];
        
        // Vary the parallel coordinate
        sibling_coords[parallel_dim] = static_cast<typename patch_index_t::offset_t>(i);
        
        // Convert back to flat index
        typename patch_index_t::offset_t sibling_offset = 0;
        typename patch_index_t::offset_t multiplier = 1;
        for (size_t d = 0; d < s_dimension; ++d) {
            sibling_offset += sibling_coords[d] * multiplier;
            multiplier *= s_fanout;
        }
        
        // Get the sibling patch ID
        auto sibling_id = patch_index_t::child_of(parent_id, sibling_offset);
        finer_ids[i] = sibling_id;
    }
    
    NeighborVariant nb;
    nb.data = typename NeighborVariant::Finer{ finer_ids };
    neighbor_neighbor_array[opposite_direction] = nb;
}
// // Helper function for finer neighbor symmetry  
// void enforce_neighbor_symmetry_finer(patch_index_t patch_id, patch_index_t finer_neighbor_id,
//                                    size_t direction, size_t opposite_direction) {
//     auto neighbor_it = find_index(finer_neighbor_id);
//     if (!neighbor_it.has_value()) {
//         return; // Neighbor doesn't exist in tree yet
//     }
    
//     linear_index_t neighbor_linear_idx = neighbor_it.value()->second;
//     auto& neighbor_neighbor_array = m_neighbors[neighbor_linear_idx];
    
//     // Always overwrite with Coarser neighbor
//     NeighborVariant nb;
//     nb.data = typename NeighborVariant::Coarser{ patch_id };
//     neighbor_neighbor_array[opposite_direction] = nb;
// }

    template <typename Fn>
    auto update_refine_flags(Fn&& fn) noexcept(
        noexcept(fn(std::declval<linear_index_t&>()))
    )
    {
        for (linear_index_t i = 0; i < m_size; ++i)
        {
            m_refine_status_buffer[i] = fn(m_linear_index_map[i]);
        }
    }

    auto apply_refine_coarsen() -> void
    {
        m_to_refine.clear();
        m_to_coarsen.clear();
        std::vector<patch_index_t> parent_patch_idx;
        for (linear_index_t i = 0; i < m_size; ++i)
        {
            const auto node_id = m_linear_index_map[i];
            if (node_id.id() == 0)
            {
                continue;
            }
            const auto parent_id = patch_index_t::parent_of(node_id);
            parent_patch_idx.push_back(parent_id);
        }
        std::sort(parent_patch_idx.begin(), parent_patch_idx.end());
        parent_patch_idx.erase(
            std::unique(parent_patch_idx.begin(), parent_patch_idx.end()),
            parent_patch_idx.end()
        );

        for (linear_index_t i = 0; i < m_size; ++i)
        {
            if (is_refine_elegible(i))
            {
                const auto node_id = m_linear_index_map[i];
                m_to_refine.push_back(node_id);
            }
        }
        for (auto parent_id : parent_patch_idx)
        {
            if (is_coarsen_elegible(parent_id))
            {
                m_to_coarsen.push_back(parent_id.id());
            }
        }
    }

    auto get_neighbors(patch_index_t const& node_id, patch_index_directon_t dir)
        -> std::optional<std::vector<patch_index_t>>
    {
        std::vector<patch_index_t> neighbor_vector;

        auto cell_it = find_index(node_id);
        assert(cell_it.has_value() && "[get_neighbors] this cell cannot be found");

#ifdef AMR_NDTREE_CHECK_NEIGHBORS
        std::cout << "[get_neighbors] node_id: " << node_id.id() << " dir: " << int(dir)
                  << "\n";
#endif

        auto direct_neighbor = patch_index_t::neighbour_at(node_id, dir);

        if (!direct_neighbor) // adjacent to boundary case
        {
#ifdef AMR_NDTREE_CHECK_NEIGHBORS
            std::cout << "  [get_neighbors] No direct neighbor (boundary case)\n";
#endif
            return std::nullopt;
        }

#ifdef AMR_NDTREE_CHECK_NEIGHBORS
        std::cout << "  [get_neighbors] Direct neighbor candidate: "
                  << direct_neighbor.value().id() << "\n";
#endif
        auto direct_neighbor_it = find_index(direct_neighbor.value().id());

        if (direct_neighbor_it.has_value()) // neighbor on same level case
        {
#ifdef AMR_NDTREE_CHECK_NEIGHBORS
            std::cout << "  [get_neighbors] Found neighbor on same level: "
                      << direct_neighbor.value().id() << "\n";
#endif
            neighbor_vector.push_back(direct_neighbor.value());
            return neighbor_vector;
        }

        auto neighbor_parent = patch_index_t::parent_of(direct_neighbor.value());
#ifdef AMR_NDTREE_CHECK_NEIGHBORS
        std::cout << "  [get_neighbors] Checking parent of direct neighbor: "
                  << neighbor_parent.id() << "\n";
#endif
        auto neighbor_parent_it = find_index(neighbor_parent.id());
        if (neighbor_parent_it.has_value()) // neighbor on lower level case
        {
#ifdef AMR_NDTREE_CHECK_NEIGHBORS
            std::cout << "  [get_neighbors] Found neighbor on lower level (parent): "
                      << neighbor_parent.id() << "\n";
#endif
            neighbor_vector.push_back(neighbor_parent);
            return neighbor_vector;
        }

        typename patch_index_t::offset_t offset0, offset1;
        switch (dir)
        {
            case patch_index_directon_t::left:
                offset0 = 1;
                offset1 = 3;
                break;
            case patch_index_directon_t::right:
                offset0 = 0;
                offset1 = 2;
                break;
            case patch_index_directon_t::bottom:
                offset0 = 0;
                offset1 = 1;
                break;
            case patch_index_directon_t::top:
                offset0 = 2;
                offset1 = 3;
                break;
            default: break;
        }
        auto child0 = patch_index_t::child_of(direct_neighbor.value(), offset0);
        auto child1 = patch_index_t::child_of(direct_neighbor.value(), offset1);
#ifdef AMR_NDTREE_CHECK_NEIGHBORS
        std::cout << "  [get_neighbors] Checking children of direct neighbor: "
                  << child0.id() << ", " << child1.id() << "\n";
#endif
        auto child_0_it = find_index(child0.id());
        auto child_1_it = find_index(child1.id());

        if (child_0_it.has_value() && child_1_it.has_value())
        {
#ifdef AMR_NDTREE_CHECK_NEIGHBORS
            std::cout << "  [get_neighbors] Found neighbor children: " << child0.id()
                      << ", " << child1.id() << "\n";
#endif
            neighbor_vector.push_back(child0);
            neighbor_vector.push_back(child1);
            return neighbor_vector;
        }

#ifdef AMR_NDTREE_CHECK_NEIGHBORS
        std::cout << "  [get_neighbors] No neighbor found (unexpected case)\n";
#endif
        assert(false && "none of the four cases in get neighbor was met");
    }

    auto balancing()
    {
        constexpr patch_index_directon_t directions[] = {
            patch_index_directon_t::left,
            patch_index_directon_t::right,
            patch_index_directon_t::top,
            patch_index_directon_t::bottom
        };

        // Refinement balancing
        for (size_t i = 0; i < m_to_refine.size(); i++)
        {
            auto cell_id         = m_to_refine[i];
            auto [coords, level] = patch_index_t::decode(cell_id.id());
#ifdef AMR_NDTREE_CHECK_BALANCING
            std::cout << "[balancing] Refinement  Checking cell " << cell_id.id()
                      << " at level " << (int)level << " coords: (" << coords[0] << ","
                      << coords[1] << ")\n";
#endif
            for (auto direction : directions)
            {
                auto neighbor_opt = get_neighbors(cell_id, direction);
                if (!neighbor_opt.has_value())
                {
#ifdef AMR_NDTREE_CHECK_BALANCING
                    std::cout << "boundary in this direction " << std::endl;
#endif
                    continue;
                }
                for (const auto& neighbor : neighbor_opt.value())
                {
                    auto [__, level_bp_neighbor] = patch_index_t::decode(neighbor.id());

                    if (level_bp_neighbor < level)
                    {
#ifdef AMR_NDTREE_CHECK_BALANCING
                        std::cout << "    [balancing] Balancing violation! Refining "
                                     "neighbor cell "
                                  << neighbor.id() << "\n";
#endif
                        if (std::find_if(
                                m_to_refine.begin(),
                                m_to_refine.end(),
                                [&](const patch_index_t& n)
                                { return n.id() == neighbor.id(); }
                            ) == m_to_refine.end())
                        {
                            m_to_refine.push_back(neighbor);
                        }
                    }
                }
            }
        }

        // Coarsening balancing
        std::vector<patch_index_t> blocks_to_remove;
        for (size_t i = 0; i < m_to_coarsen.size(); i++)
        {
            auto parent_id       = m_to_coarsen[i];
            auto [coords, level] = patch_index_t::decode(parent_id.id());
#ifdef AMR_NDTREE_CHECK_BALANCING
            std::cout << "[balancing] Coarsening Checking parent " << parent_id.id()
                      << " at level " << (int)level << " coords: (" << coords[0] << ","
                      << coords[1] << ")\n";
#endif

            for (auto direction : directions)
            {
                // For each direction, check the two children on the face
                typename patch_index_t::offset_t offset0 = 0, offset1 = 0;
                switch (direction)
                {
                    case patch_index_directon_t::left:
                        offset0 = 0;
                        offset1 = 2;
                        break;
                    case patch_index_directon_t::right:
                        offset0 = 1;
                        offset1 = 3;
                        break;
                    case patch_index_directon_t::bottom:
                        offset0 = 2;
                        offset1 = 3;
                        break;
                    case patch_index_directon_t::top:
                        offset0 = 0;
                        offset1 = 1;
                        break;
                    default: break;
                }
                std::vector<typename patch_index_t::offset_t> offsets = { offset0,
                                                                          offset1 };
                for (auto offset : offsets)
                {
                    auto child_cell = patch_index_t::child_of(parent_id.id(), offset);
                    [[maybe_unused]] auto [child_coords, child_level] =
                        patch_index_t::decode(child_cell.id());
#ifdef AMR_NDTREE_CHECK_BALANCING
                    std::cout << "  [coarsen] Checking child " << child_cell.id()
                              << " (offset " << int(offset) << ") at level "
                              << int(child_level) << " coords: (" << child_coords[0]
                              << "," << child_coords[1] << ")\n";
#endif

                    auto result = get_neighbors(child_cell, direction);
                    if (!result)
                    {
#ifdef AMR_NDTREE_CHECK_BALANCING
                        std::cout << "    [coarsen] No neighbor in direction "
                                  << int(direction) << "\n";
#endif
                        continue;
                    }
                    for (const auto& neighbor_id : *result)
                    {
                        auto [__, neighbor_id_level] =
                            patch_index_t::decode(neighbor_id.id());
                        auto iterator = std::find_if(
                            m_to_refine.begin(),
                            m_to_refine.end(),
                            [&](const patch_index_t& n)
                            { return n.id() == neighbor_id.id(); }
                        );
                        if (iterator != m_to_refine.end()) // this is untested...
                        {
                            neighbor_id_level++;
                        }
#ifdef AMR_NDTREE_CHECK_BALANCING
                        std::cout << "    [coarsen] Neighbor " << neighbor_id.id()
                                  << " at level " << int(neighbor_id_level) << "\n";
#endif
                        if (neighbor_id_level > level + 1)
                        {
#ifdef AMR_NDTREE_CHECK_BALANCING
                            std::cout
                                << "    [balancing] Balancing violation! removing this "
                                   "block from coarsening: parent "
                                << parent_id.id() << " (neighbor " << neighbor_id.id()
                                << " is finer)\n";
#endif
                            blocks_to_remove.push_back(parent_id);
                            break;
                        }
                    }
                }
            }
        }
        for (const auto& id : blocks_to_remove)
        {
            m_to_coarsen.erase(
                std::remove(m_to_coarsen.begin(), m_to_coarsen.end(), id),
                m_to_coarsen.end()
            );
        }
    }

public:
    template <typename Fn>
    auto reconstruct_tree(Fn&& fn) noexcept(noexcept(fn(std::declval<linear_index_t&>())))
    {
        update_refine_flags(fn);
        apply_refine_coarsen();
        balancing();
        fragment();
        recombine();
    }

    [[nodiscard]]
    auto get_node_index_at(linear_index_t idx) const noexcept -> patch_index_t
    {
        assert(idx < m_size && "Index out of bounds in node_index_at()");
        return m_linear_index_map[idx];
    }

private:
    [[nodiscard, gnu::always_inline]]
    auto back_idx() noexcept -> linear_index_t
    {
        return m_size - 1;
    }

    auto append(patch_index_t const node_id, neighbor_array_t neighbor_array) noexcept -> void
    {
        m_linear_index_map[m_size] = node_id;
        m_index_map[node_id]       = m_size;
        m_neighbors[m_size] = neighbor_array;
        ++m_size;
    }

    [[nodiscard]]
    auto find_index(patch_index_t const node_id) const noexcept
        -> std::optional<index_map_const_iterator_t>
    {
        const auto it = m_index_map.find(node_id);
        return it == m_index_map.end() ? std::nullopt : std::optional{ it };
    }

    [[nodiscard]]
    auto find_index(patch_index_t const node_id) noexcept
        -> std::optional<index_map_iterator_t>
    {
        const auto it = m_index_map.find(node_id);
        return it == m_index_map.end() ? std::nullopt : std::optional{ it };
    }

    // TODO: make private
public:
    auto sort_buffers() noexcept -> void
    {
        compact();
        std::sort(
            m_reorder_buffer,
            &m_reorder_buffer[m_size],
            [this](auto const i, auto const j)
            { return m_linear_index_map[i] < m_linear_index_map[j]; }
        );

        linear_index_t        backup_start_pos;
        patch_index_t         backup_node_index;
        refine_status_t       backup_refine_status;
        neighbor_array_t      backup_neighbors;
        
        // NEW: Backup buffer for entire patches instead of single elements
        using backup_patch_t = std::array<deconstructed_types_t, patch_layout_t::s_flat_size>;
        backup_patch_t backup_patch;

        for (linear_index_t i = 0; i != back_idx();)
        {
            auto src = m_reorder_buffer[i];
            if (i == src)
            {
                ++i;
                continue;
            }

            backup_start_pos     = i;
            backup_node_index    = m_linear_index_map[i];
            backup_refine_status = m_refine_status_buffer[i];
            backup_neighbors = m_neighbors[i];
            
            // Backup entire patch
            auto patch_i_start = i * patch_layout_t::s_flat_size;
            [this, &backup_patch, patch_i_start]<std::size_t... I>(std::index_sequence<I...>)
            {
                for(size_t k = 0; k < patch_layout_t::s_flat_size; k++) {
                    ((void)(std::get<I>(backup_patch[k]) = std::get<I>(m_data_buffers)[patch_i_start + k]), ...);
                }
            }(std::make_index_sequence<std::tuple_size_v<deconstructed_buffers_t>>{});

            auto dst = i;
            do
            {
                m_linear_index_map[dst]     = m_linear_index_map[src];
                m_refine_status_buffer[dst] = m_refine_status_buffer[src];
                m_neighbors[dst] = m_neighbors[src];
                
                // Copy entire patches instead of single elements
                auto patch_src_start = src * patch_layout_t::s_flat_size;
                auto patch_dst_start = dst * patch_layout_t::s_flat_size;
                
                std::apply(
                    [patch_src_start, patch_dst_start](auto&... b) { 
                        for(size_t k = 0; k < patch_layout_t::s_flat_size; k++) {
                            ((b[patch_dst_start + k] = b[patch_src_start + k]), ...);
                        }
                    }, 
                    m_data_buffers
                );
                
                m_index_map[m_linear_index_map[dst]] = dst;
                m_reorder_buffer[dst]                = dst;
                dst                                  = src;
                src                                  = m_reorder_buffer[src];
                assert(src != dst);
            } while (src != backup_start_pos);

            m_linear_index_map[dst]     = backup_node_index;
            m_refine_status_buffer[dst] = backup_refine_status;
            m_neighbors[dst] = backup_neighbors; 
            
            // Restore backed up patch
            auto patch_dst_start = dst * patch_layout_t::s_flat_size;
            [this, &backup_patch, patch_dst_start]<std::size_t... I>(std::index_sequence<I...>)
            {
                for(size_t k = 0; k < patch_layout_t::s_flat_size; k++) {
                    ((void)(std::get<I>(m_data_buffers)[patch_dst_start + k] = std::get<I>(backup_patch[k])), ...);
                }
            }(std::make_index_sequence<std::tuple_size_v<deconstructed_buffers_t>>{});
            
            m_index_map[backup_node_index] = dst;
            m_reorder_buffer[dst]          = dst;
        }
        assert(is_sorted());
        assert(std::ranges::is_sorted(m_reorder_buffer, &m_reorder_buffer[m_size]));
    }

public:
    [[nodiscard]]
    auto gather_node(linear_index_t const i) const noexcept -> value_type

    {
        return std::apply(
            [i](auto&&... args)
            { return value_type(std::forward<decltype(args)>(args)[i]...); },
            m_data_buffers
        );
    }

    auto scatter_node(value_type const& v, const linear_index_t i) const noexcept -> void
    {
        [this, &v, i]<std::size_t... I>(std::index_sequence<I...>)
        {
            ((void)(std::get<I>(m_data_buffers)[i] = std::get<I>(v.data_tuple()).value),
             ...);
        }(std::make_index_sequence<std::tuple_size_v<deconstructed_buffers_t>>{});
    }

auto restrict_patches(linear_index_t const start_from, linear_index_t const to) noexcept
    -> void
{
    std::apply(
        [to](auto&... b)
        {
            for(size_t k = 0; k < patch_layout_t::s_flat_size; k++) {
                ((b[to + k] =  std::remove_reference_t<decltype(b[0])>{}), ...); // Zero initialize
            }
        },
        m_data_buffers
    );

    for(size_t patch_idx = 0; patch_idx < s_nd_fanout; patch_idx++) {
        for(size_t linear_idx = 0; linear_idx < static_cast<size_t>(patch_layout_t::layout_t::s_logical_flat_size); linear_idx++) {

            auto map_value = s_patch_maps[static_cast<int>(patch_idx)][static_cast<int>(linear_idx)]; 

            auto full_map_value = patch_layout_t::layout_t::logical_to_full_index(map_value);
            auto full_linear_idx = patch_layout_t::layout_t::logical_to_full_index(linear_idx);

            auto parent_linear_idx = to +  full_map_value;
            auto child_linear_idx = start_from + patch_layout_t::s_flat_size * patch_idx + full_linear_idx;

            
            std::apply(
                [parent_linear_idx, child_linear_idx](auto&... b)
                {
                    ((void)(b[parent_linear_idx] += b[child_linear_idx] ), ...);
                },
                m_data_buffers
            );
        }
    }

    std::apply(
        [to](auto&... b)
        {
            for(size_t k = 0; k < patch_layout_t::s_flat_size; k++) {
                ((b[to + k] /= static_cast<std::remove_reference_t<decltype(b[0])>>(s_nd_fanout)), ...);
            }
        },
        m_data_buffers
    );
}

auto interpolate_patch(
    linear_index_t const from,
    linear_index_t const start_to
) noexcept -> void
{

    for(size_t patch_idx = 0; patch_idx < s_nd_fanout; patch_idx++) {

        for(size_t linear_idx = 0; linear_idx < static_cast<size_t>(patch_layout_t::layout_t::s_logical_flat_size); linear_idx++) {
            

            auto map_value = s_patch_maps[static_cast<int>(patch_idx)][static_cast<int>(linear_idx)];
            
            auto full_map_value = patch_layout_t::layout_t::logical_to_full_index(map_value);
            auto full_linear_idx = patch_layout_t::layout_t::logical_to_full_index(linear_idx);

            auto parent_linear_idx = from +  full_map_value;
            auto child_linear_idx = start_to + patch_layout_t::s_flat_size * patch_idx + full_linear_idx;

            std::apply(
                [parent_linear_idx, child_linear_idx, patch_idx, linear_idx](auto&... b)
                {
                    ((void)(
                        b[child_linear_idx] = b[parent_linear_idx]
                    ), ...);
                },
                m_data_buffers
            );
        }
    }
}

    auto interpolate_node(
        linear_index_t const from,
        linear_index_t const start_to
    ) const noexcept -> void
    {
        auto const old_node = gather_node(from);
        std::cout << old_node << '\n';
#ifdef AMR_NDTREE_ENABLE_CHECKS
        for (auto i = decltype(s_nd_fanout){}; i != s_nd_fanout; ++i)
        {
            assert(m_index_map.at(m_linear_index_map[start_to + i]) == start_to + i);
        }
#endif
        std::apply(
            [from, start_to](auto&... b)
            {
                // TODO: Implement
                for (auto i = decltype(s_nd_fanout){}; i != s_nd_fanout; ++i)
                {
                    ((void)(b[start_to + i] =
                                b[from] + static_cast<value_t<decltype(b)>>(i + 1)),
                     ...);
                }
            },
            m_data_buffers
        );
    }

    [[nodiscard]]
    auto get_refine_status(const linear_index_t i) const noexcept -> refine_status_t
    {
        assert(i < m_size);
        return m_refine_status_buffer[i];
    }

private:
    [[nodiscard]]
    auto is_refine_elegible(const linear_index_t i) const noexcept -> bool
    {
        const auto node_id = m_linear_index_map[i];
        assert(m_index_map.contains(node_id));
        const auto status = m_refine_status_buffer[i];
        const auto level  = patch_index_t::level(node_id);
        return (status == refine_status_t::Refine) &&
               (level < patch_index_t::max_depth());
    }

    [[nodiscard]]
    auto is_coarsen_elegible(patch_index_t parent_id) const noexcept -> bool
    {
        for (typename patch_index_t::offset_t i = 0; i < s_nd_fanout; ++i)
        {
            const auto child = patch_index_t::child_of(parent_id, i);
            const auto it    = find_index(child.id());
            // TODO: Maybe do this an assert rather
            if (!it.has_value())
            {
                return false;
            }
            const auto idx = it.value()->second;
            if (m_refine_status_buffer[idx] != refine_status_t::Coarsen)
            {
                return false;
            }
        }
        return true;
    }

    // TODO: privatize
public:
    auto compact() noexcept -> void
    {
        size_t tail = 0;
        for (linear_index_t head = 0; head < m_size; ++head)
        {
            const auto node_id = m_linear_index_map[head];
            if (m_index_map.contains(node_id))
            {
                block_buffer_swap(head, tail);
                ++tail;
            }
        }
        m_size = tail;
        m_index_map.clear();
        for (linear_index_t i = 0; i != m_size; ++i)
        {
            m_index_map[m_linear_index_map[i]] = i;
        }
    }

    [[gnu::always_inline, gnu::flatten]]
auto block_buffer_swap(linear_index_t const i, linear_index_t const j) noexcept
    -> void
{
    
    assert(i < m_size);
    assert(j < m_size);
    if (i == j)
    {
        return;
    }
    // std::cout << "switching " << i << " and " << j << "with block size "<<  patch_layout_t::s_flat_size << std::endl;
    assert(m_linear_index_map[i] != m_linear_index_map[j]);
    std::swap(m_linear_index_map[i], m_linear_index_map[j]);
    std::swap(m_refine_status_buffer[i], m_refine_status_buffer[j]);
    std::swap(m_neighbors[i], m_neighbors[j]);
    auto patch_i_start = i * patch_layout_t::s_flat_size;
    auto patch_j_start = j * patch_layout_t::s_flat_size;
    
    std::apply(
        [patch_i_start, patch_j_start](auto&... b) {
            ((void)(
                std::swap_ranges(&b[patch_i_start], 
                               &b[patch_i_start + patch_layout_t::s_flat_size], 
                               &b[patch_j_start])
            ), ...);
        }, 
        m_data_buffers
    );
}

    [[nodiscard]]
    auto is_sorted() const noexcept -> bool
    {
        if (std::ranges::is_sorted(
                m_linear_index_map, &m_linear_index_map[m_size], std::less{}
            ))
        {
            for (linear_index_t i = 0; i != m_size; ++i)
            {
                assert(m_index_map.contains(m_linear_index_map[i]));
                if (m_index_map.at(m_linear_index_map[i]) != i)
                {
                    std::cout << "index map is not correct" << std::endl;
                    return false;
                }
            }
            return true;
        }
        std::cout << "linear index is not sorted" << std::endl;
        ;
        return false;
    }

#ifdef AMR_NDTREE_ENABLE_CHECKS
    auto check_index_map() const noexcept -> void
    {
        assert(m_index_map.size() <= m_size);
        for (const auto& [node_idx, linear_idx] : m_index_map)
        {
            assert(m_linear_index_map[linear_idx] == node_idx);
        }
        // std::cout << "Hash table looks good chef...\n";
    }
#endif

private:
    index_map_t                m_index_map;
    deconstructed_buffers_t    m_data_buffers;
    linear_index_map_t         m_linear_index_map;
    linear_index_array_t       m_reorder_buffer;
    flat_refine_status_array_t m_refine_status_buffer;
    size_type                  m_size;
    std::vector<patch_index_t> m_to_refine;
    std::vector<patch_index_t> m_to_coarsen;
    neighbor_buffer_t          m_neighbors;
};

} // namespace amr::ndt::tree

#endif // AMR_INCLUDED_NDTREE
