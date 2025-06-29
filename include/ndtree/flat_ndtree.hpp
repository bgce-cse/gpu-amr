#ifndef AMR_INCLUDED_FLAT_NDTREE
#define AMR_INCLUDED_FLAT_NDTREE

#include "ndconcepts.hpp"
#include "ndhierarchy.hpp"
#include "utility/compile_time_utility.hpp"
#include "utility/constexpr_functions.hpp"
#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>

#ifndef NDEBUG
// #    define AMR_NDTREE_CHECKBOUNDS 
// #    define AMR_NDTREE_ENABLE_CHECKS
#endif

namespace amr::ndt::tree
{

template <concepts::DeconstructibleType T, concepts::NodeIndex Node_Index>
class flat_ndtree
{
private:
    enum struct RefinementStatus : char
    {
        Stable  = 0,
        Refine  = 1,
        Coarsen = 2,
    };

    using refine_status_t = RefinementStatus;

public:
    using value_type                  = T;
    using node_index_t                = Node_Index;
    using size_type                   = std::size_t;
    using flat_index_t                = size_type;
    using node_index_directon_t       = typename node_index_t::direction_t;
    static constexpr auto s_nd_fanout = node_index_t::nd_fanout();

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

    static_assert(s_nd_fanout > 1);

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
        using type = std::tuple<typename Ts::type...>;
    };

    using deconstruced_types_t =
        typename deconstructed_types_impl<typename T::deconstructed_types_map_t>::type;

    using flat_index_map_t           = pointer_t<node_index_t>;
    using flat_index_array_t         = pointer_t<flat_index_t>;
    using flat_refine_status_array_t = pointer_t<refine_status_t>;
    using index_map_t                = std::unordered_map<node_index_t, flat_index_t>;
    using index_map_iterator_t       = typename index_map_t::iterator;
    using index_map_const_iterator_t = typename index_map_t::const_iterator;

public:
    flat_ndtree(size_type size) noexcept
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
            (pointer_t<node_index_t>)std::malloc(size * sizeof(node_index_t));
        m_sort_buffer = (pointer_t<flat_index_t>)std::malloc(size * sizeof(flat_index_t));
        m_refinement_status_buffer =
            (pointer_t<refine_status_t>)std::malloc(size * sizeof(refine_status_t));

        append(node_index_t::root());
    }

    ~flat_ndtree() noexcept
    {
        std::free(m_refinement_status_buffer);
        std::free(m_sort_buffer);
        std::free(m_linear_index_map);
        std::apply([](auto&... b) { (void)(std::free(b), ...); }, m_data_buffers);
    }

public:
    [[nodiscard]]
    auto size() const noexcept -> size_type
    {
        return m_size;
    }

    template <concepts::TypeMap Map_Type>
    [[nodiscard, gnu::always_inline, gnu::flatten]]
    auto get(flat_index_t const idx) noexcept -> reference_t<typename Map_Type::type>
    {
        // assert(idx < m_size);
        return std::get<pointer_t<typename Map_Type::type>>(m_data_buffers)[idx];
    }

    template <concepts::TypeMap Map_Type>
    [[nodiscard, gnu::always_inline, gnu::flatten]]
    auto get(flat_index_t const idx) const noexcept
        -> const_reference_t<typename Map_Type::type>
    {
        // assert(idx < m_size);
        return std::get<pointer_t<typename Map_Type::type>>(m_data_buffers)[idx];
    }

    auto fragment(node_index_t const node_id) -> void
    {
        auto it = find_index(node_id);
        assert(it.has_value());
        auto const start_to = m_size;
        for (auto i = decltype(s_nd_fanout){}; i != s_nd_fanout; ++i)
        {
            auto child_id = node_index_t::child_of(node_id, i);
            assert(!find_index(child_id).has_value());
            append(child_id);
            assert(m_index_map[child_id] == back_idx());
            assert(m_linear_index_map[back_idx()] == child_id);
        }
        const auto from = it.value()->second;
        interpolate_node(from, start_to);
        block_swap(from, back_idx());
        m_index_map.erase(it.value());
        --m_size;
        assert(
            m_linear_index_map[from] == node_index_t::child_of(node_id, s_nd_fanout - 1)
        );
        assert(m_index_map[node_index_t::child_of(node_id, s_nd_fanout - 1)] == from);
#ifdef AMR_NDTREE_ENABLE_CHECKS
        check_index_map();
#endif
    }

    auto recombine(node_index_t const parent_node_id) -> void
    {
        // sort_buffers();
        std::cout << "[recombine] Start for parent " << parent_node_id.id() << "\n";

        assert(!find_index(parent_node_id).has_value());

        const auto child_0    = node_index_t::child_of(parent_node_id, 0);
        const auto child_0_it = find_index(child_0);
        assert(child_0_it.has_value());

        const auto start = child_0_it.value()->second;
        std::cout << "[recombine] Appending parent " << parent_node_id.id() << " at back_idx " << back_idx() + 1 << "\n";
        append(parent_node_id);
        assert(m_linear_index_map[back_idx()] == parent_node_id);
        std::cout << "[recombine] Restricting nodes from \n";
        restrict_nodes(start, back_idx());
        const auto max_swaps = m_size - start - s_nd_fanout;
        std::cout << "[recombine] max_swaps: " << max_swaps << ", m_size: " << m_size << ", start: " << start << ", s_nd_fanout: " << s_nd_fanout << "\n";
        // assert(max_swaps > 0);

        for (auto i = decltype(s_nd_fanout){}; i != s_nd_fanout; ++i)
        {
            const auto child_i    = node_index_t::child_of(parent_node_id, i);
            auto       child_i_it = find_index(child_i);
            assert(child_i_it.has_value());
            std::cout << "[recombine] Handling child " << i << " (id: " << child_i.id() << ") at index " << child_i_it.value()->second << "\n";
            if (i < max_swaps)
            {
                std::cout << "[recombine] Swapping child index " << child_i_it.value()->second << " with back_idx " << back_idx() << "\n";
                block_swap(child_i_it.value()->second, back_idx());
            }
            std::cout << "[recombine] Erasing child " << child_i.id() << " from index map\n";
            m_index_map.erase(child_i_it.value());
            --m_size;
            // print_linear_index_map("[recombine] Morton index array after erasure:");
        }
        std::cout << "[recombine] After loop, parent should be at index " << start << "\n";
        // print_linear_index_map("[recombine] Morton index array at end:");
        // assert(m_linear_index_map[start] == parent_node_id);
        // assert(m_index_map[parent_node_id] == start);
        std::cout << "[recombine] Done for parent " << parent_node_id.id() << "\n";
    }



    auto fragment(std::vector<node_index_t>& to_refine)
    {
        sort_buffers();
        std::cout << "[fragment] to_refine vector contains " << to_refine.size() << " entries:\n";
        for (size_t i = 0; i < to_refine.size(); ++i)
        {
            std::cout << "  [" << i << "] id = " << to_refine[i].id() << std::endl;
        }
        for (size_t idx = 0; idx < to_refine.size(); idx++)
        {
            fragment(to_refine[idx]);
            sort_buffers();
        }
    }

    auto recombine(const std::vector<node_index_t>& node_ids) -> void
    {
        sort_buffers();
        std::cout << "[recombine] to_coarsen vector contains " << node_ids.size() << " entries:\n";
        for (size_t i = 0; i < node_ids.size(); ++i)
        {
            auto [coords, level] = node_index_t::decode(node_ids[i].id());
            std::cout << "  [" << i << "] id = " << node_ids[i].id() << " coords: ("
                    << coords[0] << "," << coords[1] << ")"
                    << " level: " << static_cast<int>(level) << std::endl;
        }

        for (const auto& node_id : node_ids)
        {
            auto [coords, level] = node_index_t::decode(node_id.id());
            std::cout << "[recombine] Recombining node_id: " << node_id.id()
                    << " coords: (" << coords[0] << "," << coords[1] << ")"
                    << " level: " << static_cast<int>(level) << std::endl;
            recombine(node_id);
            sort_buffers();
        }
    }
    


    template <typename Lambda>
    void compute_refine_flag(Lambda&& condition)
    {
        for (flat_index_t i = 0; i < m_size; ++i)
        {
            auto node_id = m_linear_index_map[i];
            // The lambda should return an int or RefinementStatus (0=Stable, 1=Refine, 2=Coarsen)
            auto flag = condition(node_id);
            m_refinement_status_buffer[i] = static_cast<refine_status_t>(flag);
        }
    }
    auto apply_refine_coarsen()
        -> std::pair<std::vector<node_index_t>, std::vector<node_index_t>>
    {
        std::vector<node_index_t> to_refine;
        std::vector<node_index_t> to_coarsen;
        std::vector<node_index_t> parent_morton_idx;
        for (flat_index_t i = 0; i < m_size; ++i)
        {
            auto node_id = m_linear_index_map[i];
            if (node_id.id() == 0){continue;} 
            auto parent_id = node_index_t::parent_of(node_id);
            parent_morton_idx.push_back(parent_id);
        }
        std::sort(parent_morton_idx.begin(), parent_morton_idx.end());
        parent_morton_idx.erase(
            std::unique(parent_morton_idx.begin(), parent_morton_idx.end()),
            parent_morton_idx.end()
        );

        for (flat_index_t i = 0; i < m_size; ++i)
        {
            auto node_id = m_linear_index_map[i];
            auto status = m_refinement_status_buffer[i];
            auto [coords, level] = node_index_t::decode(node_id.id());
            if (status == refine_status_t::Refine)
            {
                if (level < node_index_t::max_depth())
                    to_refine.push_back(node_id);
            }
        }
        for (auto parent_id : parent_morton_idx)
        {
            std::cout << "[apply_refine_coarsen] Checking parent " << parent_id.id() << " for coarsening...\n";
            for (int i = 0; i < 4; ++i) {
                auto child = node_index_t::child_of(parent_id, i);
                std::cout << "  child " << i << ": " << child.id();
                auto it = find_index(child.id());
                if (it.has_value()) {
                    auto idx = it.value()->second;
                    std::cout << " (status: " << int(m_refinement_status_buffer[idx]) << ")";
                } else {
                    std::cout << " (not found)";
                }
                std::cout << "\n";
            }
            if (coarsen_all(parent_id))
            {
                std::cout << "  -> All children marked for coarsening, adding parent " << parent_id.id() << "\n";
                to_coarsen.push_back(parent_id.id());
            }
            else
            {
                std::cout << "  -> Not all children marked for coarsening.\n";
            }
        }
        return {to_refine, to_coarsen};
    }


    auto get_neighbors(node_index_t const& node_id, node_index_directon_t dir)
        -> std::optional<std::vector<node_index_t>>
    {
        std::vector<node_index_t> neighbor_vector;

        auto cell_it = find_index(node_id);
        assert(cell_it.has_value() && "[get_neighbors] this cell cannot be found");
        std::cout << "[get_neighbors] node_id: " << node_id.id() << " dir: " << int(dir) << "\n";

        auto direct_neighbor = node_index_t::neighbour_at(node_id, dir);

        if (!direct_neighbor) // adjacent to boundary case
        {
            std::cout << "  [get_neighbors] No direct neighbor (boundary case)\n";
            return std::nullopt;
        }

        std::cout << "  [get_neighbors] Direct neighbor candidate: " << direct_neighbor.value().id() << "\n";
        auto direct_neighbor_it = find_index(direct_neighbor.value().id());

        if (direct_neighbor_it.has_value()) // neighbor on same level case
        {
            std::cout << "  [get_neighbors] Found neighbor on same level: " << direct_neighbor.value().id() << "\n";
            neighbor_vector.push_back(direct_neighbor.value());
            return neighbor_vector;
        }

        auto neighbor_parent = node_index_t::parent_of(direct_neighbor.value());
        std::cout << "  [get_neighbors] Checking parent of direct neighbor: " << neighbor_parent.id() << "\n";
        auto neighbor_parent_it = find_index(neighbor_parent.id());
        if (neighbor_parent_it.has_value()) // neighbor on lower level case
        {
            std::cout << "  [get_neighbors] Found neighbor on lower level (parent): " << neighbor_parent.id() << "\n";
            neighbor_vector.push_back(neighbor_parent);
            return neighbor_vector;
        }

        typename node_index_t::offset_t offset0, offset1;
        switch (dir)
        {
            case node_index_directon_t::left:   offset0 = 1; offset1 = 3; break;
            case node_index_directon_t::right:  offset0 = 0; offset1 = 2; break;
            case node_index_directon_t::bottom: offset0 = 0; offset1 = 1; break;
            case node_index_directon_t::top:    offset0 = 2; offset1 = 3; break;
            default: break;
        }
        auto child0 = node_index_t::child_of(direct_neighbor.value(), offset0);
        auto child1 = node_index_t::child_of(direct_neighbor.value(), offset1);
        std::cout << "  [get_neighbors] Checking children of direct neighbor: "
                  << child0.id() << ", " << child1.id() << "\n";
        auto child_0_it = find_index(child0.id());
        auto child_1_it = find_index(child1.id());
        
        if (child_0_it.has_value() && child_1_it.has_value())
        {
            std::cout << "  [get_neighbors] Found neighbor children: "
                      << child0.id() << ", " << child1.id() << "\n";
            neighbor_vector.push_back(child0);
            neighbor_vector.push_back(child1);
            return neighbor_vector;
        }

        std::cout << "  [get_neighbors] No neighbor found (unexpected case)\n";
        assert(false && "none of the four cases in get neighbor was met");
    }

    auto balancing(
    std::vector<node_index_t>& to_refine,
    std::vector<node_index_t>& to_coarsen
)
{

    constexpr node_index_directon_t directions[] = {
        node_index_directon_t::left,
        node_index_directon_t::right,
        node_index_directon_t::top,
        node_index_directon_t::bottom
    };

    // Refinement balancing
    for (size_t i = 0; i < to_refine.size(); i++)
    {
        auto cell_id = to_refine[i];
        auto [coords, level] = node_index_t::decode(cell_id.id());
        std::cout << "[balancing] Refinement  Checking cell " << cell_id.id()
                  << " at level " << (int)level << " coords: (" << coords[0] << ","
                  << coords[1] << ")\n";
        for (auto direction : directions)
        {
            auto neighbor_opt = get_neighbors(cell_id, direction);
            if (!neighbor_opt.has_value())
            {
                std::cout << "boundary in this direction " << std::endl;
                continue;
            }
            for (const auto& neighbor : neighbor_opt.value())
            {
                auto [__, level_bp_neighbor] = node_index_t::decode(neighbor.id());

                if (level_bp_neighbor < level)
                {
                    std::cout << "    [balancing] Balancing violation! Refining neighbor cell "
                              << neighbor.id() << "\n";
                    if (std::find_if(
                            to_refine.begin(),
                            to_refine.end(),
                            [&](const node_index_t& n) { return n.id() == neighbor.id(); }
                        ) == to_refine.end())
                    {
                        to_refine.push_back(neighbor);
                    }
                }
            }
        }
    }

    // Coarsening balancing
    std::vector<node_index_t> blocks_to_remove;
    for (size_t i = 0; i < to_coarsen.size(); i++)
    {
        auto parent_id = to_coarsen[i];
        auto [coords, level] = node_index_t::decode(parent_id.id());
        std::cout << "[balancing] Coarsening Checking parent " << parent_id.id()
                  << " at level " << (int)level << " coords: (" << coords[0] << ","
                  << coords[1] << ")\n";

        for (auto direction : directions)
        {
            // For each direction, check the two children on the face
            typename node_index_t::offset_t offset0 = 0, offset1 = 0;
            switch (direction)
            {
                case node_index_directon_t::left:   offset0 = 0; offset1 = 2; break;
                case node_index_directon_t::right:  offset0 = 1; offset1 = 3; break;
                case node_index_directon_t::bottom: offset0 = 2; offset1 = 3; break;
                case node_index_directon_t::top:    offset0 = 0; offset1 = 1; break;
                default: break;
            }
            std::vector<typename node_index_t::offset_t> offsets = {offset0, offset1};
            for (auto offset : offsets)
            {
                auto child_cell = node_index_t::child_of(parent_id.id(), offset);
                auto [child_coords, child_level] = node_index_t::decode(child_cell.id());
                std::cout << "  [coarsen] Checking child " << child_cell.id()
                          << " (offset " << int(offset) << ") at level " << int(child_level)
                          << " coords: (" << child_coords[0] << "," << child_coords[1] << ")\n";

                auto result = get_neighbors(child_cell, direction);
                if (!result)
                {
                    std::cout << "    [coarsen] No neighbor in direction " << int(direction) << "\n";
                    continue;
                }
                for (const auto& neighbor_id : *result)
                {
                    auto [__, neighbor_id_level] = node_index_t::decode(neighbor_id.id());
                    std::cout << "    [coarsen] Neighbor " << neighbor_id.id()
                              << " at level " << int(neighbor_id_level) << "\n";
                    if (neighbor_id_level > level + 1)
                    {
                        std::cout << "    [balancing] Balancing violation! removing this "
                                     "block from coarsening: parent " << parent_id.id()
                                  << " (neighbor " << neighbor_id.id() << " is finer)\n";
                        blocks_to_remove.push_back(parent_id);
                        break;
                    }
                }
            }
        }
    }
    for (const auto& id : blocks_to_remove)
    {
        to_coarsen.erase(
            std::remove(to_coarsen.begin(), to_coarsen.end(), id), to_coarsen.end()
        );
    }
}


public :
    [[nodiscard, gnu::always_inline]]
    auto back_idx() noexcept -> flat_index_t

{
    return m_size - 1;
}

    auto append(node_index_t const node_id) noexcept -> void
    {
        m_linear_index_map[m_size] = node_id;
        m_index_map[node_id]       = m_size;
        ++m_size;
    }

    [[nodiscard]]
    auto find_index(node_index_t const node_id) const noexcept
        -> std::optional<index_map_const_iterator_t>
    {
        const auto it = m_index_map.find(node_id);
        return it == m_index_map.end() ? std::nullopt : std::optional{ it };
    }

    [[nodiscard]]
    auto find_index(node_index_t const node_id) noexcept
        -> std::optional<index_map_iterator_t>
    {
        const auto it = m_index_map.find(node_id);
        return it == m_index_map.end() ? std::nullopt : std::optional{ it };
    }

    // TODO: make private
public:

auto sort_buffers() noexcept -> void
{
    std::iota(m_sort_buffer, &m_sort_buffer[m_size], 0);
    std::sort(
        m_sort_buffer,
        &m_sort_buffer[m_size],
        [this](auto const i, auto const j)
        { return m_linear_index_map[i] < m_linear_index_map[j]; }
    );

    // Temporary buffers
    std::vector<node_index_t> tmp_index(m_size);
    std::vector<refine_status_t> tmp_status(m_size);

    // For each data buffer in m_data_buffers, create a temp vector
    auto tmp_data_buffers = std::apply(
        [this](auto&... b) {
            return std::make_tuple(std::vector<value_t<decltype(b)>>(m_size)...);
        },
        m_data_buffers
    );

    // Copy sorted data into temporaries
    for (flat_index_t i = 0; i < m_size; ++i) {
        tmp_index[i] = m_linear_index_map[m_sort_buffer[i]];
        tmp_status[i] = m_refinement_status_buffer[m_sort_buffer[i]];
    }
    copy_to_tmp_buffers(m_data_buffers, tmp_data_buffers, m_sort_buffer, m_size);

    // Copy back from temporaries
    for (flat_index_t i = 0; i < m_size; ++i) {
        m_linear_index_map[i] = tmp_index[i];
        m_refinement_status_buffer[i] = tmp_status[i];
        m_index_map[m_linear_index_map[i]] = i;
    }
    copy_from_tmp_buffers(m_data_buffers, tmp_data_buffers, m_size);
}

// Helper to copy sorted data into temporaries
template <typename TupleBuffers, typename TupleTmpBuffers, std::size_t... Is>
void copy_to_tmp_buffers_impl(
    TupleBuffers& buffers, TupleTmpBuffers& tmp_buffers,
    flat_index_t* sort_buffer, size_t m_sizee, std::index_sequence<Is...>)
{
    (..., (
        [&] {
            auto& buf = std::get<Is>(buffers);
            auto& tmp_b = std::get<Is>(tmp_buffers);
            for (size_t i = 0; i < m_sizee; ++i)
                tmp_b[i] = buf[sort_buffer[i]];
        }()
    ));
}

template <typename TupleBuffers, typename TupleTmpBuffers>
void copy_to_tmp_buffers(TupleBuffers& buffers, TupleTmpBuffers& tmp_buffers, flat_index_t* sort_buffer, size_t m_sizee) {
    constexpr std::size_t N = std::tuple_size<std::remove_reference_t<TupleBuffers>>::value;
    copy_to_tmp_buffers_impl(buffers, tmp_buffers, sort_buffer, m_sizee, std::make_index_sequence<N>{});
}

/// Helper to copy back from temporaries
template <typename TupleBuffers, typename TupleTmpBuffers, std::size_t... Is>
void copy_from_tmp_buffers_impl(
    TupleBuffers& buffers, TupleTmpBuffers& tmp_buffers,
    size_t m_sizee, std::index_sequence<Is...>)
{
    (..., (
        [&] {
            auto& buf = std::get<Is>(buffers);
            auto& tmp_b = std::get<Is>(tmp_buffers);
            for (size_t i = 0; i < m_sizee; ++i)
                buf[i] = tmp_b[i];
        }()
    ));
}

template <typename TupleBuffers, typename TupleTmpBuffers>
void copy_from_tmp_buffers(TupleBuffers& buffers, TupleTmpBuffers& tmp_buffers, size_t m_sizee) {
    constexpr std::size_t N = std::tuple_size<std::remove_reference_t<TupleBuffers>>::value;
    copy_from_tmp_buffers_impl(buffers, tmp_buffers, m_sizee, std::make_index_sequence<N>{});
}
    auto sort_buffers_old() noexcept -> void
    {
        std::iota(m_sort_buffer, &m_sort_buffer[m_size], 0);
        std::sort(
            m_sort_buffer,
            &m_sort_buffer[m_size],
            [this](auto const i, auto const j)
            { return m_linear_index_map[i] < m_linear_index_map[j]; }
        );
        std::cout << '\n';
        for (flat_index_t i = 0; i != back_idx(); ++i)
        {
            const auto j = m_sort_buffer[i];
            std::cout << "i: " << i << ", idx: " << m_linear_index_map[i].repr()
                      << "\nj: " << j << ", idx: " << m_linear_index_map[j].repr()
                      << '\n';
                      if (i == j) continue;
            block_swap(i, j);
        }
        for (flat_index_t i = 0; i != m_size; ++i)
        {
            assert(m_index_map[m_linear_index_map[i]] == i);
        }
    }

private:
    [[gnu::always_inline, gnu::flatten]]
    auto block_swap(flat_index_t const i, flat_index_t const j) noexcept -> void
    {
        if (i == j)
        {
            return;
        }
        assert(i < m_size);
        assert(j < m_size);
        assert(m_linear_index_map[i] != m_linear_index_map[j]);
        
        {
            const auto i_it = m_index_map.find(m_linear_index_map[i]);
            const auto j_it = m_index_map.find(m_linear_index_map[j]);
            assert(i_it != m_index_map.end());
            assert(j_it != m_index_map.end());
            assert(i_it != j_it);
            std::swap(i_it->second, j_it->second);
        }
        std::swap(m_linear_index_map[i], m_linear_index_map[j]);
        std::swap(m_refinement_status_buffer[i], m_refinement_status_buffer[j]);
        std::apply(
            [i, j](auto&... b) { (void)(std::swap(b[i], b[j]), ...); }, m_data_buffers
        );
    }

    [[nodiscard]]
    auto gather_node(flat_index_t const i) const noexcept -> value_type
    {
        return std::apply(
            [i](auto&&... args)
            { return value_type(std::forward<decltype(args)>(args)[i]...); },
            m_data_buffers
        );
    }

auto scatter_node(value_type const& v, const flat_index_t i) const noexcept -> void
{
    std::apply(
        [&v, i](auto&... b)
        { (void)((b[i] = std::get<value_t<decltype(b)>>(v.data_tuple()).value), ...); },
        m_data_buffers
    );
}

    auto restrict_nodes(flat_index_t const start_from, flat_index_t const to) noexcept
        -> void
    {
        std::cout << "In restriction from [" << start_from << ", "
                  << start_from + s_nd_fanout - 1 << "] to " << to << '\n';
        auto mean = [](auto const data[s_nd_fanout])
        {
            auto ret = data[0];
            for (auto i = 1u; i != s_nd_fanout; ++i)
            {
                ret += data[i];
            }
            return ret / s_nd_fanout;
        };
        std::apply(
            [start_from, to, &mean](auto&... b)
            { (void)((b[to] = mean(&(b[start_from]))), ...); },
            m_data_buffers
        );
    }

    auto interpolate_node(
        flat_index_t const from,
        flat_index_t const start_to
    ) const noexcept -> void
    {
        std::cout << "In interpolation from " << from << " to [" << start_to << ", "
                  << start_to + s_nd_fanout - 1 << "]\n";
        auto const old_node = gather_node(from);
        std::cout << old_node << '\n';
        std::apply(
            [from, start_to](auto&... b)
            {
                for (auto i = decltype(s_nd_fanout){}; i != s_nd_fanout; ++i)
                {
                    (void)((b[start_to + i] =
                                b[from] * static_cast<value_t<decltype(b)>>(i + 1)),
                           ...);
                }
            },
            m_data_buffers
        );
    }

    [[nodiscard]]
    auto get_refine_status(flat_index_t const i) const noexcept -> refine_status_t
    {
        assert(i < m_size);
        return m_refinement_status_buffer[i];
    }
public:
  [[nodiscard]]
auto node_index_at(flat_index_t idx) const noexcept -> node_index_t
{
    assert(idx < m_size && "Index out of bounds in node_index_at()");
    return m_linear_index_map[idx];
}

[[nodiscard]]
auto coarsen_all(node_index_t parent_id) const noexcept -> bool
{
    for (int i = 0; i < 4; ++i) {
        auto child = node_index_t::child_of(parent_id, i);
        auto it = find_index(child.id());
        if (!it.has_value()) {
            // Child not found, cannot coarsen
            return false;
        }
        flat_index_t idx = it.value()->second;
        if (m_refinement_status_buffer[idx] != refine_status_t::Coarsen) {
            return false;
        }
    }
    return true;
}



public:


void print_linear_index_map(const std::string& msg = "") const
{
    if (!msg.empty()) std::cout << msg << "\n";
    std::cout << "m_linear_index_map [size=" << m_size << "]: ";
    for (size_t i = 0; i < m_size; ++i)
        std::cout << m_linear_index_map[i].id() << " ";
    std::cout << "\n";
}

void compact()
{
    size_t write = 0;
    for (size_t read = 0; read < m_size; ++read) {
        auto node_id = m_linear_index_map[read];
        if (m_index_map.count(node_id)) { // Only keep valid nodes
            if (write != read) {
                m_linear_index_map[write] = m_linear_index_map[read];
                m_refinement_status_buffer[write] = m_refinement_status_buffer[read];
                std::apply([&](auto&... b) { (void)(std::swap(b[write], b[read]), ...); }, m_data_buffers);
                m_index_map[node_id] = write;
            }
            ++write;
        }
    }
    m_size = write;
}



#ifdef AMR_NDTREE_ENABLE_CHECKS
    auto check_index_map() const noexcept -> void
    {
        assert(m_index_map.size() == m_size);
        std::cout << "Index map: " << '\n';
        for (const auto& [key, value] : m_index_map)
        {
            std::cout << "Key:[" << key.repr() << "] Value:[" << value << "]\n";
        }
        for (flat_index_t i = 0; i != m_size; ++i)
        {
            std::cout << m_linear_index_map[i].repr() << '\n';
            assert(m_index_map.at(m_linear_index_map[i]) == i);
        }
        std::cout << "Hash table looks good chef...\n";
    }
#endif

private:
    index_map_t                m_index_map;
    deconstructed_buffers_t    m_data_buffers;
    flat_index_map_t           m_linear_index_map;
    flat_index_array_t         m_sort_buffer;
    flat_refine_status_array_t m_refinement_status_buffer;
    size_type                  m_size;
};

} // namespace amr::ndt::tree

#endif // AMR_INCLUDED_FLAT_NDTREE
