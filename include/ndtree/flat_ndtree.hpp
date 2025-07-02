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

template <concepts::DeconstructibleType T, concepts::NodeIndex Node_Index>
class flat_ndtree
{
public:
    using value_type                  = T;
    using node_index_t                = Node_Index;
    using size_type                   = std::size_t;
    using linear_index_t              = size_type;
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

    using linear_index_map_t         = pointer_t<node_index_t>;
    using linear_index_array_t       = pointer_t<linear_index_t>;
    using flat_refine_status_array_t = pointer_t<refine_status_t>;
    using index_map_t                = std::unordered_map<node_index_t, linear_index_t>;
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
        m_reorder_buffer =
            (pointer_t<linear_index_t>)std::malloc(size * sizeof(linear_index_t));
        m_refine_status_buffer =
            (pointer_t<refine_status_t>)std::malloc(size * sizeof(refine_status_t));
        std::iota(m_reorder_buffer, &m_reorder_buffer[size], 0);

        append(node_index_t::root());
    }

    ~flat_ndtree() noexcept
    {
        std::free(m_refine_status_buffer);
        std::free(m_reorder_buffer);
        std::free(m_linear_index_map);
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

    auto fragment(node_index_t const node_id) -> void
    {
        const auto it = find_index(node_id);
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
        m_index_map.erase(it.value());
#ifdef AMR_NDTREE_ENABLE_CHECKS
        check_index_map();
#endif
    }

    auto recombine(node_index_t const parent_node_id) -> void
    {
        assert(!find_index(parent_node_id).has_value());

        const auto child_0    = node_index_t::child_of(parent_node_id, 0);
        const auto child_0_it = find_index(child_0);
        assert(child_0_it.has_value());

        const auto start = child_0_it.value()->second;
        append(parent_node_id);
        assert(m_linear_index_map[back_idx()] == parent_node_id);
        restrict_nodes(start, back_idx());

        for (auto i = decltype(s_nd_fanout){}; i != s_nd_fanout; ++i)
        {
            const auto child_i    = node_index_t::child_of(parent_node_id, i);
            auto       child_i_it = find_index(child_i);
            assert(child_i_it.has_value());
            assert(child_i_it.value()->second == start + i);
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
        std::vector<node_index_t> parent_morton_idx;
        for (linear_index_t i = 0; i < m_size; ++i)
        {
            const auto node_id = m_linear_index_map[i];
            if (node_id.id() == 0)
            {
                continue;
            }
            const auto parent_id = node_index_t::parent_of(node_id);
            parent_morton_idx.push_back(parent_id);
        }
        std::sort(parent_morton_idx.begin(), parent_morton_idx.end());
        parent_morton_idx.erase(
            std::unique(parent_morton_idx.begin(), parent_morton_idx.end()),
            parent_morton_idx.end()
        );

        for (linear_index_t i = 0; i < m_size; ++i)
        {
            if (is_refine_elegible(i))
            {
                const auto node_id = m_linear_index_map[i];
                m_to_refine.push_back(node_id);
            }
        }
        for (auto parent_id : parent_morton_idx)
        {
            if (is_coarsen_elegible(parent_id))
            {
                m_to_coarsen.push_back(parent_id.id());
            }
        }
    }

    auto get_neighbors(node_index_t const& node_id, node_index_directon_t dir)
        -> std::optional<std::vector<node_index_t>>
    {
        std::vector<node_index_t> neighbor_vector;

        auto cell_it = find_index(node_id);
        assert(cell_it.has_value() && "[get_neighbors] this cell cannot be found");

#ifdef AMR_NDTREE_CHECK_NEIGHBORS
        std::cout << "[get_neighbors] node_id: " << node_id.id() << " dir: " << int(dir)
                  << "\n";
#endif

        auto direct_neighbor = node_index_t::neighbour_at(node_id, dir);

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

        auto neighbor_parent = node_index_t::parent_of(direct_neighbor.value());
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

        typename node_index_t::offset_t offset0, offset1;
        switch (dir)
        {
            case node_index_directon_t::left:
                offset0 = 1;
                offset1 = 3;
                break;
            case node_index_directon_t::right:
                offset0 = 0;
                offset1 = 2;
                break;
            case node_index_directon_t::bottom:
                offset0 = 0;
                offset1 = 1;
                break;
            case node_index_directon_t::top:
                offset0 = 2;
                offset1 = 3;
                break;
            default: break;
        }
        auto child0 = node_index_t::child_of(direct_neighbor.value(), offset0);
        auto child1 = node_index_t::child_of(direct_neighbor.value(), offset1);
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
        constexpr node_index_directon_t directions[] = { node_index_directon_t::left,
                                                         node_index_directon_t::right,
                                                         node_index_directon_t::top,
                                                         node_index_directon_t::bottom };

        // Refinement balancing
        for (size_t i = 0; i < m_to_refine.size(); i++)
        {
            auto cell_id         = m_to_refine[i];
            auto [coords, level] = node_index_t::decode(cell_id.id());
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
                    auto [__, level_bp_neighbor] = node_index_t::decode(neighbor.id());

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
                                [&](const node_index_t& n)
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
        std::vector<node_index_t> blocks_to_remove;
        for (size_t i = 0; i < m_to_coarsen.size(); i++)
        {
            auto parent_id       = m_to_coarsen[i];
            auto [coords, level] = node_index_t::decode(parent_id.id());
#ifdef AMR_NDTREE_CHECK_BALANCING
            std::cout << "[balancing] Coarsening Checking parent " << parent_id.id()
                      << " at level " << (int)level << " coords: (" << coords[0] << ","
                      << coords[1] << ")\n";
#endif

            for (auto direction : directions)
            {
                // For each direction, check the two children on the face
                typename node_index_t::offset_t offset0 = 0, offset1 = 0;
                switch (direction)
                {
                    case node_index_directon_t::left:
                        offset0 = 0;
                        offset1 = 2;
                        break;
                    case node_index_directon_t::right:
                        offset0 = 1;
                        offset1 = 3;
                        break;
                    case node_index_directon_t::bottom:
                        offset0 = 2;
                        offset1 = 3;
                        break;
                    case node_index_directon_t::top:
                        offset0 = 0;
                        offset1 = 1;
                        break;
                    default: break;
                }
                std::vector<typename node_index_t::offset_t> offsets = { offset0,
                                                                         offset1 };
                for (auto offset : offsets)
                {
                    auto child_cell = node_index_t::child_of(parent_id.id(), offset);
                    [[maybe_unused]] auto [child_coords, child_level] =
                        node_index_t::decode(child_cell.id());
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
                            node_index_t::decode(neighbor_id.id());
                        auto iterator = std::find_if(
                            m_to_refine.begin(),
                            m_to_refine.end(),
                            [&](const node_index_t& n)
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
    auto get_node_index_at(linear_index_t idx) const noexcept -> node_index_t
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
        compact();
        std::sort(
            m_reorder_buffer,
            &m_reorder_buffer[m_size],
            [this](auto const i, auto const j)
            { return m_linear_index_map[i] < m_linear_index_map[j]; }
        );

        linear_index_t        backup_start_pos;
        node_index_t          backup_node_index;
        refine_status_t       backup_refine_status;
        deconstructed_types_t backup_buffer;

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
            [this, &backup_buffer, i]<std::size_t... I>(std::index_sequence<I...>)
            {
                ((void)(std::get<I>(backup_buffer) = std::get<I>(m_data_buffers)[i]),
                 ...);
            }(std::make_index_sequence<std::tuple_size_v<deconstructed_buffers_t>>{});

            auto dst = i;
            do
            {
                m_linear_index_map[dst]     = m_linear_index_map[src];
                m_refine_status_buffer[dst] = m_refine_status_buffer[src];
                std::apply(
                    [src, dst](auto&... b) { ((void)(b[dst] = b[src]), ...); },
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
            [this, &backup_buffer, dst]<std::size_t... I>(std::index_sequence<I...>)
            {
                ((void)(std::get<I>(m_data_buffers)[dst] = std::get<I>(backup_buffer)),
                 ...);
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

    auto restrict_nodes(linear_index_t const start_from, linear_index_t const to) noexcept
        -> void
    {
        // std::cout << "In restriction from [" << start_from << ", "
        //           << start_from + s_nd_fanout - 1 << "] to " << to << '\n';
        auto mean = [](auto const data[s_nd_fanout])
        {
            auto ret = data[0];
            for (auto i = 1u; i != s_nd_fanout; ++i)
            {
                ret += data[i] / s_nd_fanout;
            }
            return ret;
        };
        std::apply(
            [start_from, to, &mean](auto&... b)
            { ((void)(b[to] = mean(&(b[start_from]))), ...); },
            m_data_buffers
        );
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
        const auto level  = node_index_t::level(node_id);
        return (status == refine_status_t::Refine) && (level < node_index_t::max_depth());
    }

    [[nodiscard]]
    auto is_coarsen_elegible(node_index_t parent_id) const noexcept -> bool
    {
        for (typename node_index_t::offset_t i = 0; i < s_nd_fanout; ++i)
        {
            const auto child = node_index_t::child_of(parent_id, i);
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
        assert(m_linear_index_map[i] != m_linear_index_map[j]);
        std::swap(m_linear_index_map[i], m_linear_index_map[j]);
        std::swap(m_refine_status_buffer[i], m_refine_status_buffer[j]);
        std::apply(
            [i, j](auto&... b) { ((void)std::swap(b[i], b[j]), ...); }, m_data_buffers
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
        std::cout << "Hash table looks good chef...\n";
    }
#endif

private:
    index_map_t                m_index_map;
    deconstructed_buffers_t    m_data_buffers;
    linear_index_map_t         m_linear_index_map;
    linear_index_array_t       m_reorder_buffer;
    flat_refine_status_array_t m_refine_status_buffer;
    size_type                  m_size;
    std::vector<node_index_t>  m_to_refine;
    std::vector<node_index_t>  m_to_coarsen;
};

} // namespace amr::ndt::tree

#endif // AMR_INCLUDED_FLAT_NDTREE
