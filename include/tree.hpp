#ifndef AMR_INCLUDED_FLAT_NDTREE
#define AMR_INCLUDED_FLAT_NDTREE

#include "data_types.hpp"
#include "ndconcepts.hpp"
#include "refinement_kernels.hpp"
#include "tree_types.hpp"
#include "utility/compile_time_utility.hpp"
#include "utility/constexpr_functions.hpp"
#include "utility/error_handling.hpp"
#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <ranges>
#include <set>
#include <span>
#include <system_error>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>

#ifndef NDEBUG
#define AMR_NDTREE_CHECKBOUNDS
#define AMR_NDTREE_ENABLE_CHECKS
#endif

template <
    typename Domain, concepts::DeconstructibleType T,
    concepts::NodeIndex Node_Index>
class ndtree
{
  public:
    using value_type = T;
    using node_index_t = Node_Index;
    using size_type = std::size_t;
    using linear_index_t = size_type;
    using node_index_direction_t = typename node_index_t::direction_t;
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

    using deconstructed_buffers_t = typename deconstructed_buffers_impl<
        typename T::deconstructed_types_map_t>::type;

    template <typename>
    struct deconstructed_types_impl;

    template <typename... Ts>
        requires concepts::detail::type_map_tuple_impl<std::tuple<Ts...>>
    struct deconstructed_types_impl<std::tuple<Ts...>>
    {
        using type = std::tuple<value_t<typename Ts::type>...>;
    };

    using deconstructed_types_t = typename deconstructed_types_impl<
        typename T::deconstructed_types_map_t>::type;

    static constexpr auto s_buffer_width_v =
        std::tuple_size_v<deconstructed_buffers_t>;

    enum struct RefinementStatus : char
    {
        Stable = 0,
        Refine = 1,
        Coarsen = 2,
    };

    struct neighbours_t
    {
        struct sibling
        {
            node_index_t value;
        };
        struct uncle
        {
            node_index_t value;
        };
        struct nephews
        {
            std::array<node_index_t, s_nd_fanout> values;
        };
        using alternatves_t = std::variant<sibling, uncle, nephews>;
    };
    using neighbour_alternatives_t = typename neighbours_t::alternatves_t;

    using refine_status_t = RefinementStatus;

    using linear_index_map_t = pointer_t<node_index_t>;
    using linear_index_array_t = pointer_t<linear_index_t>;
    using flat_refine_status_array_t = pointer_t<refine_status_t>;
    using index_map_t = std::unordered_map<node_index_t, linear_index_t>;
    using index_map_iterator_t = typename index_map_t::iterator;
    using index_map_const_iterator_t = typename index_map_t::const_iterator;
    using cell_type_array = pointer_t<cell_type>;

  public:
    /**
     * @brief Constructs and initializes a tree within a Domain.
     *
     * @param domain Domain that containts the tree. Provides context
     * information and injects interpolation functions.
     * @param capacity Maximum number of nodes in the tree.
     * @param init_kernel Initiral refinemenr kernel that applies an initial
     * refinement to the tree.
     * @param init_value Initial value for all the nodes in the tree.
     */
    ndtree(
        pointer_t<Domain> domain, size_type capacity,
        concepts::InitKernel auto init_kernel, value_type init_value
    ) noexcept
        : m_size{}
        , m_domain(domain)
    {
        std::apply(
            [capacity](auto&... b) {
                ((void)(b = (pointer_t<value_t<decltype(b)>>)std::malloc(
                            capacity * sizeof(value_t<decltype(b)>)
                        )),
                 ...);
            },
            m_data_buffers
        );
        m_linear_index_map = (pointer_t<node_index_t>)std::malloc(
            capacity * sizeof(node_index_t)
        );
        m_reorder_buffer = (pointer_t<linear_index_t>)std::malloc(
            capacity * sizeof(linear_index_t)
        );
        std::iota(m_reorder_buffer, &m_reorder_buffer[capacity], 0);
        m_refine_status_buffer = (pointer_t<refine_status_t>)std::malloc(
            capacity * sizeof(refine_status_t)
        );
        m_cell_types =
            (pointer_t<cell_type>)std::malloc(capacity * sizeof(cell_type));

        append(node_index_t::root(), cell_type::DEFAULT);

        while (init_kernel.reapply())
        {
            reconstruct_tree(init_kernel);
        }

        [this, &init_value]<std::size_t... I>(std::index_sequence<I...>) {
            ((void)(std::fill_n(
                 std::get<I>(m_data_buffers),
                 m_size,
                 std::get<I>(init_value.data_tuple()).value
             )),
             ...);
        }(std::make_index_sequence<s_buffer_width_v>{});

        set_initial_cell_types(init_kernel.pgm_data());
    }

    ~ndtree() noexcept
    {
        std::free(m_refine_status_buffer);
        std::free(m_reorder_buffer);
        std::free(m_linear_index_map);
        std::apply(
            [](auto&... b) { ((void)std::free(b), ...); }, m_data_buffers
        );
    }

  public:
    /**
     * @brief Accessor to the number of leaf nodes in the tree.
     *
     * @return Number of leaf nodes in the tree.
     */
    [[nodiscard]]
    auto size() const noexcept -> size_type
    {
        return m_size;
    }

    /**
     * @brief Non-const accessor to a QOI of a fluid cell of the tree. Accessing
     * a non fluid element is undefined.
     *
     * @param Map_Type QOI to access.
     * @param morton_id Node id
     * @return Mutable reference to the underlying QOI requested
     */
    template <concepts::TypeMap Map_Type>
    [[nodiscard, gnu::always_inline, gnu::flatten]]
    auto get(node_index_t const morton_id
    ) noexcept -> reference_t<typename Map_Type::type>
    {
        const auto linear_index = index_at(morton_id);
        return std::get<Map_Type::index()>(m_data_buffers)[linear_index];
    }

    /**
     * @brief Const accessor to a QOI of a fluid cell of the tree. Accessing
     * a non fluid element is undefined.
     *
     * @param Map_Type QOI to access.
     * @param morton_id Node id
     * @return Const reference to the underlying QOI requested
     */
    template <concepts::TypeMap Map_Type>
    [[nodiscard, gnu::always_inline, gnu::flatten]]
    auto get(node_index_t const morton_id
    ) const noexcept -> const_reference_t<typename Map_Type::type>
    {
        assert(m_cell_types[m_index_map.at(morton_id)] == cell_type::FLUID);
        const auto linear_index = index_at(morton_id);
        return std::get<Map_Type::index()>(m_data_buffers)[linear_index];
    }

    /**
     * @brief Acceessor to a QUI of a leaf or parent node. Accessing a non-leaf
     * or non-parent node is undefined.
     *
     * @param morton_id Node id
     * @return Possibly manufactured of a leaf or parent node.
     */
    template <concepts::TypeMap Map_Type>
    [[nodiscard, gnu::always_inline, gnu::flatten]]
    auto get_value_recur(node_index_t const& morton_id
    ) const noexcept -> value_t<typename Map_Type::type>
    {
        const auto it = find_index(morton_id);
        if (it)
        {
            assert(m_cell_types[it.value()->second] == cell_type::FLUID);
            return get<Map_Type>(it.value()->second);
        }
        return m_domain->template restrict_impl<Map_Type>(
            std::views::iota(decltype(s_nd_fanout){}, s_nd_fanout) |
            std::views::transform([morton_id](const auto i) {
                return node_index_t::child_of(morton_id, i);
            }) |
            std::views::filter([this](auto const node_id) {
                auto const cit = find_index(node_id);
                return !cit.has_value() ||
                       m_cell_types[cit.value()->second] == cell_type::FLUID;
            }) |
            std::views::transform([this](auto const& node_id) {
                return get_value_recur<Map_Type>(node_id);
            })
        );
    }

    /**
     * @brief Mutable acceessor to a QUI of an element of the underlying
     * container. If the index is greater than or equal to the size of the tree
     * then beahviour is undefined.
     *
     * @param morton_id Element id
     * @return Reference to the underlying buffer value
     */
    template <concepts::TypeMap Map_Type>
    [[nodiscard, gnu::always_inline, gnu::flatten]]
    auto get(linear_index_t const idx
    ) noexcept -> reference_t<typename Map_Type::type>
    {
        assert(m_cell_types[idx] == cell_type::FLUID);
        assert(idx < m_size);
        return std::get<Map_Type::index()>(m_data_buffers)[idx];
    }

    /**
     * @brief Const acceessor to a QUI of an element of the underlying
     * container. If the index is greater than or equal to the size of the tree
     * then beahviour is undefined.
     *
     * @param morton_id Element id
     * @return Const reference to the underlying buffer value
     */
    template <concepts::TypeMap Map_Type>
    [[nodiscard, gnu::always_inline, gnu::flatten]]
    auto get(linear_index_t const idx
    ) const noexcept -> const_reference_t<typename Map_Type::type>
    {
        assert(m_cell_types[idx] == cell_type::FLUID);
        assert(idx < m_size);
        return std::get<Map_Type::index()>(m_data_buffers)[idx];
    }

    /**
     * @brief Accessor to the underlying buffer index of a leaf node.
     *
     * @param node_id Leaf node index.
     * @return Underlying buffer index of the corresponding element.
     */
    [[nodiscard]]
    auto index_at(node_index_t const node_id) const noexcept -> linear_index_t
    {
        const auto it = find_index(node_id);
        assert(it);
        return it.value()->second;
    }

    /**
     * @brief Cell type accessor
     *
     * @param idx Underlying buffer index.
     * @return Cell type at the provided index.
     */
    [[nodiscard]]
    auto get_cell_type(linear_index_t idx) const -> cell_type const&
    {
        return m_cell_types[idx];
    }

    /**
     * @brief Cell type accessor
     *
     * @param idx Leaf node index.
     * @return Cell type at the provided index.
     */
    [[nodiscard]]
    auto get_cell_type(node_index_t idx) const -> cell_type const&
    {
        return m_cell_types[m_index_map.at(idx)];
    }

    /**
     * @brief Initializes a fragmented tree given a matching topology matrix.
     *
     * @param p_topology Pointer to the topology matrix
     */
    auto set_initial_cell_types(Matrix<cell_type> const* const p_topology
    ) -> void
    {
        for (linear_index_t idx = 0; idx < m_size; idx++)
        {
            auto coords = node_index_t::rel_coords(m_linear_index_map[idx]);
            auto size = p_topology->num_rows();
            const auto i = static_cast<unsigned>(
                std::round(coords[0] * static_cast<float>(size))
            );
            const auto j = static_cast<unsigned>(
                std::round(coords[1] * static_cast<float>(size))
            );
            m_cell_types[idx] = (*p_topology)(i, j);
            if (m_cell_types[idx] == cell_type::OBSTACLE)
            {
                std::apply(
                    [idx](auto&... buffers) { ((buffers[idx] = 0.0), ...); },
                    m_data_buffers
                );
            }
        }
    }

    /**
     * @brief Fragments a leaf node
     *
     * @param node_id Node to fragment. New parent.
     */
    auto fragment(node_index_t const node_id) -> void
    {
        const auto it = find_index(node_id);
        assert(it.has_value());
        auto const start_to = m_size;
        for (auto i = decltype(s_nd_fanout){}; i != s_nd_fanout; ++i)
        {
            auto child_id = node_index_t::child_of(node_id, i);
            assert(!find_index(child_id).has_value());
            append(child_id, m_cell_types[m_index_map[node_id]]);
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

    /**
     * @brief Recombines a set of leaves that share an immediate parent.
     *
     * @param parent_node_id Common parent node id.
     */
    auto recombine(node_index_t const parent_node_id) -> void
    {
        assert(!find_index(parent_node_id).has_value());

        const auto child_0 = node_index_t::child_of(parent_node_id, 0);
        const auto child_0_it = find_index(child_0);
        assert(child_0_it.has_value());

        const auto start = child_0_it.value()->second;
        assert(is_coarsen_elegible(parent_node_id));

        append(parent_node_id, m_cell_types[m_index_map[child_0]]);
        assert(m_linear_index_map[back_idx()] == parent_node_id);
        restrict_nodes(start, back_idx());

        for (auto i = decltype(s_nd_fanout){}; i != s_nd_fanout; ++i)
        {
            const auto child_i = node_index_t::child_of(parent_node_id, i);
            auto child_i_it = find_index(child_i);
            assert(child_i_it.has_value());
            assert(child_i_it.value()->second == start + i);
            m_index_map.erase(child_i_it.value());
        }
    }

    /**
     * @brief Applies the fragmentation routine to all nodes selected for
     * fragmentation.
     */
    auto fragment() -> void
    {
        assert(is_sorted());
        for (auto& node_id : m_to_refine)
        {
            fragment(node_id);
        }
        sort_buffers();
    }

    /**
     * @brief Applies the refinement routine to all nodes selected for
     * refinement.
     */
    auto recombine() -> void
    {
        assert(is_sorted());
        for (const auto& node_id : m_to_coarsen)
        {
            // if(!is_coarsen_elegible(node_id)){continue;}
            recombine(node_id);
        }
        sort_buffers();
    }

    /**
     * @brief Updates the refinement flags given a decision kernel.
     *
     * @param fn Decision kernel.
     */
    template <typename Fn>
    auto update_refine_flags(Fn&& fn
    ) noexcept(noexcept(fn(std::declval<linear_index_t&>()))) -> void
    {
        for (linear_index_t i = 0; i < m_size; ++i)
        {
            m_refine_status_buffer[i] = fn(m_linear_index_map[i]);
        }
    }

    /**
     * @brief Collects indices of refine and coarsen elegible nodes.
     */
    auto apply_refine_coarsen() -> void
    {
        m_to_refine.clear();
        m_to_coarsen.clear();
        std::vector<node_index_t> parent_morton_idx;
        for (linear_index_t i = 0; i < m_size; ++i)
        {
            const auto node_id = m_linear_index_map[i];
            if (node_id == node_index_t::root())
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
                if (!std::ranges::contains(m_to_refine, node_id))
                {
                    m_to_refine.push_back(node_id);
                }
            }
        }
        for (auto parent_id : parent_morton_idx)
        {
            if (is_coarsen_elegible(parent_id))
            {
                if (!std::ranges::contains(m_to_coarsen, parent_id))
                {
                    m_to_coarsen.push_back(parent_id);
                }
            }
        }
    }

    /**
     * @brief Get leaf node neighbour ids of a node in a certain direction if
     * they exist.
     *
     * @param node_id Reference node
     * @param dir Direction in which to search for neighbours.
     * @return Optional of the possible neighbour alternatives.
     */
    [[nodiscard]]
    auto get_neighbors(node_index_t const& node_id, node_index_direction_t dir)
        const -> std::optional<neighbour_alternatives_t>
    {
        const auto sibling = node_index_t::neighbour_at(node_id, dir);
        if (!sibling) // adjacent to boundary case
        {
            return std::nullopt;
        }

        const auto sibling_it = find_index(sibling.value());
        if (sibling_it.has_value()) // neighbor on same level case
        {
            return typename neighbours_t::sibling{sibling.value()};
        }

        const auto uncle = node_index_t::parent_of(sibling.value());
        const auto uncle_it = find_index(uncle);
        if (uncle_it.has_value()) // neighbor on lower level case
        {
            return typename neighbours_t::uncle{uncle};
        }
        typename neighbours_t::nephews nephews;
        for (typename node_index_t::offset_t i = 0; i != s_nd_fanout; ++i)
        {
            nephews.values[i] = node_index_t::child_of(sibling.value(), i);
        }
        return nephews;
    }

    /**
     * @brief Ensure the efinement flags meet the balancing restrictions.
     */
    auto balancing() -> void
    {
        static constexpr node_index_direction_t directions[] = {
            node_index_direction_t::N,
            node_index_direction_t::NE,
            node_index_direction_t::E,
            node_index_direction_t::SE,
            node_index_direction_t::S,
            node_index_direction_t::SW,
            node_index_direction_t::W,
            node_index_direction_t::NW
        };

        // Refinement balancing
        for (linear_index_t i = 0; i != m_to_refine.size(); ++i)
        {
            auto const cell_id = m_to_refine[i];
            for (const auto d : directions)
            {
                const auto neighbours = get_neighbors(cell_id, d);
                if (!neighbours.has_value())
                {
                    continue;
                }
                using uncle_t = typename neighbours_t::uncle;
                auto const& neighbour = neighbours.value();
                if (std::holds_alternative<uncle_t>(neighbour))
                {
                    const auto n = std::get<uncle_t>(neighbour).value;
                    if (!std::ranges::contains(m_to_refine, n))
                    {
                        m_to_refine.push_back(n);
                    }
                }
            }
        }

        // sort m_to_refine from low level to high level
        std::sort(
            m_to_refine.begin(),
            m_to_refine.end(),
            [](const node_index_t& a, const node_index_t& b) {
                return node_index_t::level(a) < node_index_t::level(b);
            }
        );
        // Replace std::vector with unordered_set to avoid duplicates
        std::unordered_set<node_index_t> blocks_to_remove;
        for (size_type i = 0; i != m_to_coarsen.size(); ++i)
        {
            auto const parent_id = m_to_coarsen[i];

            bool blocked = false;
            for (auto direction : directions)
            {
                for (typename node_index_t::offset_t j = 0; j != s_nd_fanout;
                     ++j)
                {
                    const auto child_cell =
                        node_index_t::child_of(parent_id, j);

                    const auto neighbours =
                        get_neighbors(child_cell, direction);
                    if (!neighbours) continue;
                    using nephews_t = typename neighbours_t::nephews;
                    using sibling_t = typename neighbours_t::sibling;
                    if (std::holds_alternative<nephews_t>(neighbours.value()))
                    {
                        blocks_to_remove.insert(parent_id);
                        blocked = true;
                        break;
                    }
                    if (std::holds_alternative<sibling_t>(neighbours.value()))
                    {
                        const auto sibling =
                            std::get<sibling_t>(neighbours.value()).value;
                        if (std::ranges::contains(m_to_refine, sibling))
                        {
                            blocks_to_remove.insert(parent_id);
                            blocked = true;
                            break;
                        }
                    }
                }
                if (blocked) break;
            }
        }
        // Remove all blocked parents from m_to_coarsen
        for (auto id : blocks_to_remove)
        {
            m_to_coarsen.erase(
                std::remove(m_to_coarsen.begin(), m_to_coarsen.end(), id),
                m_to_coarsen.end()
            );
        }
    }

  public:
    /**
     * @brief Reconstruct the tree provided a refinement decission kernel.
     *
     * @param fn Refinement decission kernel.
     */
    template <typename Fn>
    auto reconstruct_tree(Fn&& fn
    ) noexcept(noexcept(fn(std::declval<linear_index_t&>()))) -> void
    {
        std::cout << "[reconstruct_tree] Calling update_refine_flags..."
                  << std::endl;
        update_refine_flags(fn);
        std::cout << "[reconstruct_tree] Calling apply_refine_coarsen..."
                  << std::endl;
        apply_refine_coarsen();
        std::cout << "[reconstruct_tree] Calling balancing..." << std::endl;
        balancing();
        std::cout << "[reconstruct_tree] Calling fragment..." << std::endl;
        fragment();
        std::cout << "[reconstruct_tree] Calling recombine..." << std::endl;
        recombine();
        std::cout << "[reconstruct_tree] Finished! No. of refined cells: "
                  << m_to_refine.size() << " and No. of coarsen blocks "
                  << m_to_coarsen.size() << std::endl;
    }

    /**
     * @brief Node index accessor of he corresponding underlying buffer element.
     *
     * @param idx Index of the underlying buffer. If the index is greater than
     * or equal to the size the behaviour is undefined.
     * @return
     */
    [[nodiscard]]
    auto get_node_index_at(linear_index_t idx) const noexcept -> node_index_t
    {
        assert(idx < m_size && "Index out of bounds in node_index_at()");
        return m_linear_index_map[idx];
    }

  private:
    /**
     * @brief Return index of the last leaf stored in the underlying buffer.
     *
     * @return Underlying buffer index.
     */
    [[nodiscard, gnu::always_inline]]
    auto back_idx() noexcept -> linear_index_t

    {
        return m_size - 1;
    }

    /**
     * @brief Append a new leaf to the back of the underlying buffer.
     *
     * @param node_id Index of the new node.
     * @param type Cell type to assign.
     */
    auto append(node_index_t const node_id, cell_type type) noexcept -> void
    {
        m_cell_types[m_size] = type;
        m_linear_index_map[m_size] = node_id;
        m_index_map[node_id] = m_size;
        ++m_size;
    }

  public:
    /**
     * @brief Retrunrs optional const iterator to the index table containing the
     * index into the underlying buffer of the asscoaited element.
     *
     * @param node_id Node idnex to search for.
     * @return Optional iterator into index table.
     */
    [[nodiscard]]
    auto find_index(node_index_t const node_id
    ) const noexcept -> std::optional<index_map_const_iterator_t>
    {
        const auto it = m_index_map.find(node_id);
        return it == m_index_map.end() ? std::nullopt : std::optional{it};
    }

    /**
     * @brief Retrunrs optional iterator to the index table containing the index
     * into the underlying buffer of the asscoaited element.
     *
     * @param node_id Node idnex to search for.
     * @return Optional iterator into index table.
     */
    [[nodiscard]]
    auto find_index(node_index_t const node_id
    ) noexcept -> std::optional<index_map_iterator_t>
    {
        const auto it = m_index_map.find(node_id);
        return it == m_index_map.end() ? std::nullopt : std::optional{it};
    }

  private:
    /**
     * @brief Sorts the underling buffers according to the ordering of the
     * node_index_t. Requires that the tree is compact, the beahviour is
     * undefined if else.
     */
    auto sort_buffers() noexcept -> void
    {
        compact();
        std::sort(
            m_reorder_buffer,
            &m_reorder_buffer[m_size],
            [this](auto const i, auto const j) {
                return m_linear_index_map[i] < m_linear_index_map[j];
            }
        );

        linear_index_t backup_start_pos;
        node_index_t backup_node_index;
        refine_status_t backup_refine_status;
        deconstructed_types_t backup_buffer;
        cell_type backup_cell_type;

        for (linear_index_t i = 0; i != back_idx();)
        {
            auto src = m_reorder_buffer[i];
            if (i == src)
            {
                ++i;
                continue;
            }

            backup_start_pos = i;
            backup_node_index = m_linear_index_map[i];
            backup_refine_status = m_refine_status_buffer[i];
            backup_cell_type = m_cell_types[i];
            [this,
             &backup_buffer,
             i]<std::size_t... I>(std::index_sequence<I...>) {
                ((void)(std::get<I>(backup_buffer) =
                            std::get<I>(m_data_buffers)[i]),
                 ...);
            }(std::make_index_sequence<s_buffer_width_v>{});

            auto dst = i;
            do
            {
                m_linear_index_map[dst] = m_linear_index_map[src];
                m_refine_status_buffer[dst] = m_refine_status_buffer[src];
                m_cell_types[dst] = m_cell_types[src];
                std::apply(
                    [src, dst](auto&... b) { ((void)(b[dst] = b[src]), ...); },
                    m_data_buffers
                );
                m_index_map[m_linear_index_map[dst]] = dst;
                m_reorder_buffer[dst] = dst;
                dst = src;
                src = m_reorder_buffer[src];
                assert(src != dst);
            } while (src != backup_start_pos);

            m_linear_index_map[dst] = backup_node_index;
            m_refine_status_buffer[dst] = backup_refine_status;
            m_cell_types[dst] = backup_cell_type;
            [this,
             &backup_buffer,
             dst]<std::size_t... I>(std::index_sequence<I...>) {
                ((void)(std::get<I>(m_data_buffers)[dst] =
                            std::get<I>(backup_buffer)),
                 ...);
            }(std::make_index_sequence<s_buffer_width_v>{});
            m_index_map[backup_node_index] = dst;
            m_reorder_buffer[dst] = dst;
        }
        assert(is_sorted());
        assert(
            std::ranges::is_sorted(m_reorder_buffer, &m_reorder_buffer[m_size])
        );
    }

  public:
    /**
     * @brief Constructs a node from the deconstructed values in the underlying
     * buffers. If the index is greater than or queal to the size, the behaviour
     * is undefined.
     *
     * @param i Index into the underlying buffer.
     * @return The cell stored at the respective index.
     */
    [[nodiscard]]
    auto gather_node(linear_index_t const i) const noexcept -> value_type

    {
        return std::apply(
            [i](auto&&... args) {
                return value_type(std::forward<decltype(args)>(args)[i]...);
            },
            m_data_buffers
        );
    }

    /**
     * @brief Scaters a node into the undrlying buffers by deconstructing it.
     *
     * @param v Node to deconstruct
     * @param i Tagret position in the underlying buffers to store the
     * deconstructed node.
     */
    auto scatter_node(value_type const& v, const linear_index_t i)
        const noexcept -> void
    {
        [this, &v, i]<std::size_t... I>(std::index_sequence<I...>) {
            ((void)(std::get<I>(m_data_buffers)[i] =
                        std::get<I>(v.data_tuple()).value),
             ...);
        }(std::make_index_sequence<s_buffer_width_v>{});
    }

    /**
     * @brief Creates the value of a node out of a group of contiguous siblings.
     *
     * @param start_from Index of the first of a group of sibling contiguous in
     * th eunderlying buffers.
     * @param to Index to the underlying buffers where to write the output.
     */
    auto restrict_nodes(
        linear_index_t const start_from, linear_index_t const to
    ) noexcept -> void
    {
        [this, start_from, to]<std::size_t... I>(std::index_sequence<I...>) {
            ((void)(std::get<I>(m_data_buffers)[to] =
                        m_domain->template restrict_impl<std::tuple_element_t<
                            I,
                            typename T::deconstructed_types_map_t>>(
                            std::span(
                                &std::get<I>(m_data_buffers)[start_from],
                                s_nd_fanout
                            )
                        )),
             ...);
        }(std::make_index_sequence<s_buffer_width_v>{});
    }

    /**
     * @brief Fills the value of a group of siblings from the value of another
     * node and possibly its neighbours.
     *
     * @param from Index into the underying buffer of a node.
     * @param start_to ndex into the underlying buffer to the first of a
     * contiguous set of siblings.
     */
    auto interpolate_node(
        linear_index_t const from, linear_index_t const start_to
    ) const noexcept -> void
    {
        const auto node_id = m_linear_index_map[from];
        if (m_cell_types[from] == cell_type::FLUID)
        {
            for (auto i = decltype(s_nd_fanout){}; i != s_nd_fanout; ++i)
            {
                [this,
                 start_to,
                 from,
                 i,
                 node_id]<std::size_t... I>(std::index_sequence<I...>) {
                    ((void)(std::get<I>(m_data_buffers)[start_to + i] =
                                m_domain->template interpolate_impl<
                                    std::tuple_element_t<
                                        I,
                                        typename T::deconstructed_types_map_t>>(
                                    node_id, i
                                )),
                     ...);
                }(std::make_index_sequence<s_buffer_width_v>{});
                m_cell_types[start_to + i] = m_cell_types[from];
            }
        }
        else
        {
            std::apply(
                [from, start_to](auto&... b) {
                    for (auto i = decltype(s_nd_fanout){}; i != s_nd_fanout;
                         ++i)
                    {
                        ((void)(b[start_to + i] = b[from]), ...);
                    }
                },
                m_data_buffers
            );
            for (auto i = decltype(s_nd_fanout){}; i != s_nd_fanout; ++i)
            {
                m_cell_types[start_to + i] = m_cell_types[from];
            }
        }
    }

    /**
     * @brief Access the refine status in the underlying buffer.
     *
     * @param i Index
     * @return Refine status at a position in the underlying buffer
     */
    [[nodiscard]]
    auto get_refine_status(const linear_index_t i
    ) const noexcept -> refine_status_t
    {
        assert(i < m_size);
        return m_refine_status_buffer[i];
    }

  private:
    /**
     * @brief Checks if a node can refine or not.
     *
     * @param i Node in question
     * @return If the node can refine
     */
    [[nodiscard]]
    auto is_refine_elegible(const linear_index_t i) const noexcept -> bool
    {
        const auto node_id = m_linear_index_map[i];
        assert(m_index_map.contains(node_id));
        const auto status = m_refine_status_buffer[i];
        const auto level = node_index_t::level(node_id);
        return (status == refine_status_t::Refine) &&
               (level < node_index_t::max_depth());
    }

    /**
     * @brief Checks if a group of siblings can refine.
     *
     * @param i Common parent of the siblings.
     * @return If the node can corasen
     */
    [[nodiscard]]
    auto is_coarsen_elegible(node_index_t parent_id) const noexcept -> bool
    {
        const auto c0 = node_index_t::child_of(parent_id, 0);
        if (!m_index_map.contains(c0))
        {
            return false;
        }
        const auto cell_type = m_cell_types[m_index_map.at(c0)];
        for (typename node_index_t::offset_t i = 0; i < s_nd_fanout; ++i)
        {
            const auto child = node_index_t::child_of(parent_id, i);
            const auto it = find_index(child);

            if (!it.has_value())
            {
                return false;
            }
            const auto idx = it.value()->second;
            if (m_refine_status_buffer[idx] != refine_status_t::Coarsen)
            {
                return false;
            }
            if (m_cell_types[idx] != cell_type)
            {
                std::cout << "mismatch in cell types\n";
                return false;
            }
        }
        return true;
    }

  private:
    /**
     * @brief Compacts the underying buffers.
     */
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

    /**
     * @brief Block buffer swap.
     *
     * @param i Index i
     * @param j Index j
     */
    [[gnu::always_inline, gnu::flatten]]
    auto block_buffer_swap(
        linear_index_t const i, linear_index_t const j
    ) noexcept -> void
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
        std::swap(m_cell_types[i], m_cell_types[j]);
        std::apply(
            [i, j](auto&... b) { ((void)std::swap(b[i], b[j]), ...); },
            m_data_buffers
        );
    }

    /**
     * @brief Checks if the underlying buffer is sorted.
     *
     * @return Whether the underlying buffers are sorted
     */
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

  private:
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
    index_map_t m_index_map;
    deconstructed_buffers_t m_data_buffers;
    linear_index_map_t m_linear_index_map;
    linear_index_array_t m_reorder_buffer;
    flat_refine_status_array_t m_refine_status_buffer;
    cell_type_array m_cell_types;
    size_type m_size;
    std::vector<node_index_t> m_to_refine;
    std::vector<node_index_t> m_to_coarsen;
    pointer_t<Domain> m_domain;
};

#endif // AMR_INCLUDED_FLAT_NDTREE
