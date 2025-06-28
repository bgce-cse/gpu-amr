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
        assert(!find_index(parent_node_id).has_value());

        const auto child_0    = node_index_t::child_of(parent_node_id, 0);
        const auto child_0_it = find_index(child_0);
        assert(child_0_it.has_value());

        const auto start = child_0_it.value()->second;
        append(parent_node_id);
        assert(m_linear_index_map[back_idx()] == parent_node_id);
        restrict_nodes(start, back_idx());
        const auto max_swaps = m_size - start - s_nd_fanout;
        assert(max_swaps > 0);
        for (auto i = decltype(s_nd_fanout){}; i != s_nd_fanout; ++i)
        {
            const auto child_i    = node_index_t::child_of(parent_node_id, i);
            auto       child_i_it = find_index(child_i);
            assert(child_i_it.has_value());
            if (i < max_swaps)
            {
                block_swap(child_i_it.value()->second, back_idx());
            }
            m_index_map.erase(child_i_it.value());
            --m_size;
        }
        assert(m_linear_index_map[start] == parent_node_id);
        assert(m_index_map[parent_node_id] == start);
#ifdef AMR_NDTREE_ENABLE_CHECKS
        check_index_map();
#endif
    }

private:
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
        -> std::optional<index_map_iterator_t>
    {
        const auto it = m_index_map.find(node_id);
        return it == m_index_map.end() ? std::nullopt : std::optional{ it };
    }

    [[nodiscard]]
    auto find_index(node_index_t const node_id) noexcept
        -> std::optional<index_map_const_iterator_t>
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
        std::cout << '\n';
        for (flat_index_t i = 0; i != back_idx(); ++i)
        {
            const auto j = m_sort_buffer[i];
            std::cout << "i: " << i << ", idx: " << m_linear_index_map[i].repr()
                      << "\nj: " << j << ", idx: " << m_linear_index_map[j].repr()
                      << '\n';
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
        assert(i < m_size);
        assert(j < m_size);
        assert(m_linear_index_map[i] != m_linear_index_map[j]);
        if (i == j)
        {
            return;
        }
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
            {
                (void)((b[i] = std::get<value_t<decltype(b)>>(v.data_tuple()).value),
                       ...);
            },
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

private:
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
