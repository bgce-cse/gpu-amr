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
#    define AMR_NDTREE_CHECKBOUNDS
#    define AMR_NDTREE_ENABLE_CHECKS
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
    using index_map_t                = std::unordered_map<node_index_t, flat_index_t>;
    using index_map_iterator_t       = typename index_map_t::iterator;
    using index_map_const_iterator_t = typename index_map_t::const_iterator;

public:
    flat_ndtree(size_type size) noexcept
        : m_size{}

    {
        m_linear_index_map =
            (pointer_t<node_index_t>)std::malloc(size * sizeof(node_index_t));
        m_sort_buffer = (pointer_t<flat_index_t>)std::malloc(size * sizeof(flat_index_t));
        std::apply(
            [size](auto&... b)
            {
                ((void)(b = (pointer_t<value_t<decltype(b)>>)
                            std::malloc(size * sizeof(value_t<decltype(b)>))),
                 ...);
            },
            m_data_buffers
        );
        append(node_index_t::root());
    }

    ~flat_ndtree() noexcept
    {
        std::apply([](auto&... b) { (void)(std::free(b), ...); }, m_data_buffers);
        std::free(m_sort_buffer);
        std::free(m_linear_index_map);
    }

public:
    [[nodiscard]]
    auto size() const noexcept -> size_type
    {
        return m_size;
    }

    template <concepts::TypeMap Map_Type>
    [[nodiscard, gnu::always_inline, gnu::flatten]]
    auto get(flat_index_t const& idx) noexcept -> reference_t<typename Map_Type::type>
    {
        return std::get<pointer_t<typename Map_Type::type>>(m_data_buffers)[idx];
    }

    template <concepts::TypeMap Map_Type>
    [[nodiscard, gnu::always_inline, gnu::flatten]]
    auto get(flat_index_t const& idx
    ) const noexcept -> const_reference_t<typename Map_Type::type>
    {
        return std::get<pointer_t<typename Map_Type::type>>(m_data_buffers)[idx];
    }

    [[nodiscard]]
    auto fragment(node_index_t const& node_id) -> flat_index_t
    {
        auto it = find_index(node_id);
        assert(it.has_value());
        auto const start_to = m_size;
        for (auto i = decltype(s_nd_fanout){}; i != s_nd_fanout; ++i)
        {
            auto child_id = node_index_t::child_of(node_id, i);
            assert(!find_index(child_id).has_value());
            append(child_id);
        }
        const auto from = it.value()->second;
        interpolate_node(from, start_to);
        block_swap(from, back_idx());
        m_index_map.erase(it.value()->first);
        --m_size;
        return start_to;
    }

    [[nodiscard]]
    auto recombine(node_index_t const& node_id) -> flat_index_t
    {
        assert(!find_index(node_id).has_value());

        const auto child_0    = node_index_t::child_of(node_id, 0);
        const auto child_0_it = find_index(child_0);
        assert(child_0_it.has_value());

        const auto start = child_0_it.value()->second;
        append(node_id);
        restrict_nodes(start, back_idx());
        const auto max_swaps = m_size - start - s_nd_fanout;
        for (auto i = decltype(s_nd_fanout){}; i != s_nd_fanout; ++i)
        {
            const auto child_i    = node_index_t::child_of(node_id, i);
            auto       child_i_it = find_index(child_i);
            assert(child_i_it.has_value());
            assert(child_i_it.value()->second == start + i);
            if (i < max_swaps)
            {
                block_swap(start + i, back_idx());
            }
            m_index_map.erase(child_i_it.value()->first);
            --m_size;
        }
        return start;
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
            [[maybe_unused]]
            auto _ = fragment(to_refine[idx]);
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
            [[maybe_unused]]
            auto _ = recombine(node_id);
            sort_buffers();
        }
    }


    auto get_neighbors(node_index_t const& node_id, node_index_directon_t dir)
        -> std::optional<std::vector<node_index_t>>
    {
        std::vector<node_index_t> neighbor_vector;

        auto cell_it = find_index(node_id)
        assert(cell_it.has_value() && "[get_neighbors] this cell cannot be found");

        auto direct_neighbor = node_index_t::neighbour_at(node_id, dir);

        if (!direct_neighbor) // adjacent to boundary case
        {
            return std::nullopt;
        }

        auto direct_neighbor_it = find_index(direct_neighbor.id());

        if (direct_neighbor_it.has_value()) // neighbor on same level case
        {
            neighbor_vector.push_back(direct_neighbor.value());
            return neighbor_vector;
        }

        auto neighbor_parent = node_index_t::parent_of(direct_neighbor.value());

        auto neighbor_parent_it = find_index(neighbor_parent.id());
        if (neighbor_parent_it.has_value()) // neighbor on lower level case
        {
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
                offset0 = 2;
                offset1 = 3;
                break;
            case node_index_directon_t::top:
                offset0 = 0;
                offset1 = 1;
                break;
            default: break;
        }
        auto child0 = node_index_t::child_of(direct_neighbor.value(), offset0);
        auto child1 = node_index_t::child_of(direct_neighbor.value(), offset1);
        auto child_0_it = find_index(child0.id());
        auto child_1_it = find_index(child1.id());
        
        if (child_0_it.has_value() && child_1_it.has_value())
        {
            neighbor_vector.push_back(child0);
            neighbor_vector.push_back(child1);
            return neighbor_vector;
        }

        assert(false && "none of the fours cases in get neighbor was met");
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
            auto neighbor_vector = get_neighbors(cell_id, direction);
            auto neighbor = neighbor_vector[0];
        
            auto [__, level_bp_neighbor] = node_index_t::decode(neighbor.id());
            if (level_bp_neighbor < level - 1)
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

    // Coarsening balancing
    for (size_t i = 0; i < to_coarsen.size(); i++)
    {
        auto parent_id = to_coarsen[i];
        auto [coords, level] = node_index_t::decode(parent_id.id());

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
            std::vector<int> offsets = {offset0, offset1};
            for (auto offset : offsets)
            {
                auto child_cell = node_index_t::child_of(parent_id.id(), offset);
                auto result = get_neighbors(child_cell, direction);
                if (!result)
                    continue;
                for (const auto& neighbor_id : result)
                {
                    auto [__, neighbor_id_level] = node_index_t::decode(neighbor_id.id());
                    if (neighbor_id_level > level)
                    {
                        std::cout << "    [balancing] Balancing violation! removing this "
                                     "block from coarsening "
                                  << std::endl;
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

auto append(node_index_t const& node_id) noexcept -> void
{
    m_linear_index_map[m_size] = node_id;
    m_index_map[node_id]       = m_size;
    ++m_size;
}

[[nodiscard]]
auto find_index(node_index_t const& node_id
) const noexcept -> std::optional<index_map_iterator_t>
{
    const auto it = m_index_map.find(node_id);
    return it == m_index_map.end() ? std::nullopt : std::optional{ it };
}

[[nodiscard]]
auto find_index(node_index_t const& node_id
) noexcept -> std::optional<index_map_const_iterator_t>
{
    const auto it = m_index_map.find(node_id);
    return it == m_index_map.end() ? std::nullopt : std::optional{ it };
}

auto sort_buffers() noexcept -> void
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
        std::cout << "i: " << i << ", idx: " << m_linear_index_map[i].id() << "\nj: " << j
                  << ", idx: " << m_linear_index_map[j].id() << '\n';
        block_swap(i, j);
    }
    for (flat_index_t i = 0; i != m_size; ++i)
    {
        assert(m_index_map[m_linear_index_map[i]] == i);
    }
}

[[gnu::always_inline, gnu::flatten]]
auto block_swap(flat_index_t const& i, flat_index_t const& j) noexcept -> void
{
    assert(i < m_size);
    assert(j < m_size);
    if (i == j)
    {
        return;
    }
    const auto i_it = find_index(m_linear_index_map[i]);
    const auto j_it = find_index(m_linear_index_map[j]);
    assert(i_it.has_value());
    assert(j_it.has_value());
    std::swap(m_index_map[i_it.value()->first], m_index_map[j_it.value()->first]);
    std::swap(m_linear_index_map[i], m_linear_index_map[j]);
    std::apply(
        [i, j](auto&... b) { (void)(std::swap(b[i], b[j]), ...); }, m_data_buffers
    );
}

[[nodiscard]]
auto gather_node(flat_index_t const& i) const noexcept -> value_type
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

auto restrict_nodes(flat_index_t const& start_from, flat_index_t const& to) const noexcept
    -> void
{
    std::cout << "In restriction from [" << start_from << ", "
              << start_from + s_nd_fanout - 1 << "] to " << to << '\n';
    auto mean = [](auto data[s_nd_fanout])
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
        { (void)((b[to] = mean(&b[start_from])), ...); },
        m_data_buffers
    );
}

auto interpolate_node(flat_index_t const& from, flat_index_t const& start_to)
    const noexcept -> void
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

public:
deconstructed_buffers_t m_data_buffers;
flat_index_map_t        m_linear_index_map;
flat_index_array_t      m_sort_buffer;
index_map_t             m_index_map;
size_type               m_size;
};

} // namespace amr::ndt::tree

#endif // AMR_INCLUDED_FLAT_NDTREE
