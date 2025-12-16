#ifndef AMR_INCLUDED_NDTREE
#define AMR_INCLUDED_NDTREE

#include "ndconcepts.hpp"
#include "ndtype_traits.hpp"
#include "ndutils.hpp"
#include "neighbor.hpp"
#include "patch.hpp"
#include "patch_utils.hpp"
#include "utility/compile_time_utility.hpp"
#include "utility/error_handling.hpp"
#include "utility/logging.hpp"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <ranges>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>

#ifndef NDEBUG
#    define AMR_NDTREE_CHECKBOUNDS
#    define AMR_NDTREE_ENABLE_CHECKS
// #    define AMR_NDTREE_DEBUG_PRINT_NEIGHBORS
// #    define AMR_NDTREE_DEBUG_PRINT_BALANCING
#endif

namespace amr::ndt::tree
{

template <
    concepts::DeconstructibleType T,
    concepts::PatchIndex          Patch_Index,
    concepts::PatchLayout         Patch_Layout>
class ndtree
{
public:
    using map_type      = T;
    using size_type     = std::size_t;
    using patch_index_t = Patch_Index;
    // TODO: This should be provided by the patch_index
    // using patch_index_directon_t = typename patch_index_t::direction_t;
    using linear_index_t    = size_type;
    using patch_layout_t    = Patch_Layout;
    using neighbor_utils_t  = neighbors::neighbor_utils<patch_index_t, patch_layout_t>;
    using patch_direction_t = typename neighbor_utils_t::direction_t;
    template <typename Identifier_Type>
    using neighbor_variant_base_t =
        typename neighbor_utils_t::template neighbor_variant_base_t<Identifier_Type>;
    using neighbor_patch_index_variant_t  = neighbor_variant_base_t<patch_index_t>;
    using neighbor_linear_index_variant_t = neighbor_variant_base_t<linear_index_t>;
    using patch_neighbors_t               = typename neighbor_utils_t::patch_neighbors_t;
    using halo_exchange_operator_impl_t =
        utils::patches::halo_exchange_impl_t<patch_index_t, patch_layout_t>;
    static_assert(std::is_same_v<
                  typename neighbor_utils_t::neighbor_variant_t,
                  neighbor_patch_index_variant_t>);

private:
    static constexpr size_type s_halo_width = patch_layout_t::halo_width();
    static constexpr size_type s_1d_fanout  = patch_index_t::fanout();
    static constexpr size_type s_nd_fanout  = patch_index_t::nd_fanout();
    static constexpr size_type s_rank  = patch_layout_t::rank();

    static_assert(s_1d_fanout > 1);
    static_assert(s_nd_fanout > 1);
    static_assert(
        utils::patches::multiples_of(patch_layout_t::data_layout_t::sizes(), s_1d_fanout),
        "All patch dimensions must be multiples of the fanout"
    );

    template <typename Type>
    using value_t = std::remove_pointer_t<std::remove_cvref_t<Type>>;
    template <typename Type>
    using unwrap_value_t = typename value_t<Type>::value_type;
    template <typename Type>
    using pointer_t = Type*;
    template <typename Type>
    using const_pointer_t = Type const*;
    template <typename Type>
    using reference_t = Type&;
    template <typename Type>
    using const_reference_t = Type const&;

public:
    template <typename Map_Type>
    using patch_t = patches::patch<typename Map_Type::type, patch_layout_t>;

    using deconstructed_raw_map_types_t = typename T::deconstructed_types_map_t;
    static_assert(concepts::detail::map_type_tuple_impl<deconstructed_raw_map_types_t>);
    using deconstructed_patch_types_map_t =
        type_traits::tuple_type_apply_t<patch_t, deconstructed_raw_map_types_t>;

    using deconstructed_buffers_t =
        type_traits::tuple_type_apply_t<pointer_t, deconstructed_patch_types_map_t>;
    using deconstructed_types_t =
        type_traits::tuple_type_apply_t<value_t, deconstructed_patch_types_map_t>;

public:
    enum struct RefinementStatus : std::int8_t
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
    using neighbor_buffer_t          = pointer_t<patch_neighbors_t>;

    using index_map_iterator_t       = typename index_map_t::iterator;
    using index_map_const_iterator_t = typename index_map_t::const_iterator;

private:
    static constexpr auto s_fragmentation_patch_maps =
        amr::ndt::utils::patches::fragmentation_patch_maps<patch_layout_t, s_1d_fanout>();

public:
    [[nodiscard]]
    static constexpr auto rank() noexcept -> auto const&
    {
        return s_rank;
    }

    [[nodiscard]]
    static constexpr auto nd_fanout() noexcept -> auto const&
    {
        return s_nd_fanout;
    }

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
        m_neighbors =
            (pointer_t<patch_neighbors_t>)std::malloc(size * sizeof(patch_neighbors_t));

        std::iota(m_reorder_buffer, &m_reorder_buffer[size], 0);
        patch_neighbors_t root_neighbors;
        for (auto d = patch_direction_t::first(); d != patch_direction_t::sentinel();
             d.advance())
        {
            neighbor_patch_index_variant_t periodic_neighbor;
            periodic_neighbor.data =
                typename neighbor_patch_index_variant_t::same{ patch_index_t::root() };
            root_neighbors[d.index()] = periodic_neighbor;
        }
        append(patch_index_t::root(), root_neighbors);
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

    // template <concepts::TypeMap Map_Type>
    // [[nodiscard, gnu::always_inline, gnu::flatten]]
    // auto get(linear_index_t const idx) noexcept -> reference_t<typename Map_Type::type>
    // {
    //     assert(idx < m_size);
    //     return std::get<Map_Type::index()>(m_data_buffers)[idx];
    // }

    // template <concepts::TypeMap Map_Type>
    // [[nodiscard, gnu::always_inline, gnu::flatten]]
    // auto get(linear_index_t const idx) const noexcept
    //     -> const_reference_t<typename Map_Type::type>
    // {
    //     assert(idx < m_size);
    //     return std::get<Map_Type::index()>(m_data_buffers)[idx];
    // }

    template <concepts::MapType Map_Type>
    [[nodiscard, gnu::always_inline, gnu::flatten]]
    auto get_patch(linear_index_t const linear_index) noexcept -> patch_t<Map_Type>&
    {
        return const_cast<patch_t<Map_Type>&>(
            std::as_const(*this).template get_patch<Map_Type>(linear_index)
        );
    }

    template <concepts::MapType Map_Type>
    [[nodiscard, gnu::always_inline, gnu::flatten]]
    auto get_patch(linear_index_t const linear_index) const noexcept
        -> patch_t<Map_Type> const&
    {
        assert(linear_index < m_size);
        return std::get<Map_Type::index()>(m_data_buffers)[linear_index];
    }

    template <concepts::MapType Map_Type>
    [[nodiscard, gnu::always_inline, gnu::flatten]]
    auto get_patch(patch_index_t const patch_idx) noexcept -> patch_t<Map_Type>&
    {
        return const_cast<patch_t<Map_Type>&>(
            std::as_const(*this).template get_patch<Map_Type>(patch_idx)
        );
    }

    template <concepts::MapType Map_Type>
    [[nodiscard, gnu::always_inline, gnu::flatten]]
    auto get_patch(patch_index_t const patch_idx) const noexcept
        -> patch_t<Map_Type> const&
    {
        const auto linear_index = m_index_map.at(patch_idx);
        assert(linear_index < m_size);
        return std::get<Map_Type::index()>(m_data_buffers)[linear_index];
    }

    [[nodiscard, gnu::always_inline, gnu::flatten]]
    auto get_neighbor_at(
        linear_index_t const     idx,
        patch_direction_t const& d
    ) const noexcept -> neighbor_patch_index_variant_t
    {
        assert(idx < m_size);
        return m_neighbors[idx][d.index()];
    }

    [[nodiscard]]
    auto neighbor_linear_index(
        neighbor_patch_index_variant_t const& neighbor
    ) const noexcept -> neighbor_linear_index_variant_t
    {
        using ret_t = neighbor_linear_index_variant_t;
        return std::visit(
            utils::overloads{
                [](typename neighbor_patch_index_variant_t::none const&)
                {
                    // TODO: Handle boundaries
                    // utility::error_handling::assert_unreachable();
                    return ret_t{};
                },
                [this](typename neighbor_patch_index_variant_t::same const& n)
                { return ret_t{ typename ret_t::same{ get_linear_index_at(n.id) } }; },
                [this](typename neighbor_patch_index_variant_t::finer const& ns)
                {
                    static constexpr auto K = ret_t::finer::num_neighbors();
                    return ret_t{ typename ret_t::finer{
                        utility::compile_time_utility::array_factory<linear_index_t, K>(
                            [this, &ns](auto const i)
                            { return get_linear_index_at(ns.ids[i]); }
                        ) } };
                },
                [this](typename neighbor_patch_index_variant_t::coarser const& n)
                {
                    return ret_t{
                        typename ret_t::coarser{ get_linear_index_at(n.id), n.dim_offset }
                    };
                } },
            neighbor.data
        );
    }

    auto fragment(patch_index_t const node_id) -> void
    {
        DEFAULT_SOURCE_LOG_TRACE("Fragmenting node " + node_id.repr());
        const auto it   = find_index(node_id);
        const auto from = it.value()->second;
        assert(it.has_value());
        const auto start_to = m_size;
        for (size_type i = 0; i != s_nd_fanout; ++i)
        {
            const auto child_id = patch_index_t::child_of(
                node_id, static_cast<typename patch_index_t::offset_t>(i)
            );
            assert(!find_index(child_id).has_value());

            const auto neighbor_array =
                neighbor_utils_t::compute_child_neighbors(node_id, m_neighbors[from], i);
            append(child_id, neighbor_array);
            assert(m_index_map[child_id] == back_idx());
            assert(m_linear_index_map[back_idx()] == child_id);
        }
        enforce_symmetry_after_refinement(node_id);
        interpolate_patch(from, start_to);
        m_index_map.erase(it.value());
#ifdef AMR_NDTREE_ENABLE_CHECKS
        check_index_map();
#endif
    }

    auto recombine(patch_index_t const parent_node_id) -> void
    {
        using offset_t = typename patch_index_t::offset_t;
        assert(!find_index(parent_node_id).has_value());

        const auto child_0    = patch_index_t::child_of(parent_node_id, 0);
        const auto child_0_it = find_index(child_0);
        assert(child_0_it.has_value());

        std::array<patch_neighbors_t, s_nd_fanout> child_neighbor_arrays{};
        for (auto i = offset_t{}; i != offset_t{ s_nd_fanout }; ++i)
        {
            const auto child_i    = patch_index_t::child_of(parent_node_id, i);
            const auto child_i_it = find_index(child_i);
            assert(child_i_it.has_value());
            // assert(child_i_it.value()->second == start + i);
            child_neighbor_arrays[i] = m_neighbors[m_index_map.at(child_i)];
            m_index_map.erase(child_i_it.value());
        }

        patch_neighbors_t neighbor_array =
            neighbor_utils_t::compute_parent_neighbors(child_neighbor_arrays);
        const auto start = child_0_it.value()->second;
        const auto to    = m_size;
        append(parent_node_id, neighbor_array);
        assert(m_linear_index_map[back_idx()] == parent_node_id);
        restrict_patches(start, to);

        enforce_symmetry_after_recombining(parent_node_id, neighbor_array);
    }

    auto fragment() -> void
    {
        assert(is_sorted());
        for (auto i = m_to_refine.size(); i > 0; --i)
        {
            fragment(m_to_refine[i - 1]);
        }
        sort_buffers();
        assert(is_sorted());
    }

    auto recombine() -> void
    {
        assert(is_sorted());
        for (const auto& node_id : m_to_coarsen)
        {
            recombine(node_id);
        }
        sort_buffers();
        assert(is_sorted());
    }

    template <typename Fn>
    auto update_refine_flags(Fn&& fn) noexcept(
        noexcept(fn(std::declval<linear_index_t&>()))
    )
    {
        for (linear_index_t i = 0; i != m_size; ++i)
        {
            m_refine_status_buffer[i] = fn(m_linear_index_map[i]);
        }
    }

    auto apply_refine_coarsen() noexcept -> void
    {
        m_to_refine.clear();
        m_to_coarsen.clear();
        std::vector<patch_index_t> parent_patch_idx;
        for (linear_index_t i = 0; i != m_size; ++i)
        {
            const auto node_id = m_linear_index_map[i];
            if (node_id.id() == 0)
            {
                continue;
            }
            const auto parent_id = patch_index_t::parent_of(node_id);
            parent_patch_idx.push_back(parent_id);
        }
        std::ranges::sort(parent_patch_idx, std::less{});
        parent_patch_idx.erase(
            std::unique(parent_patch_idx.begin(), parent_patch_idx.end()),
            parent_patch_idx.end()
        );

        for (linear_index_t i = 0; i != m_size; ++i)
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

    auto enforce_symmetry_after_refinement(patch_index_t parent_id) noexcept -> void
    {
        auto parent_neighbor = m_neighbors[m_index_map.at(parent_id)];

        for (auto d = patch_direction_t::first(); d != patch_direction_t::sentinel();
             d.advance())
        {
            auto& neighbor = parent_neighbor[d.index()];

            std::visit(
                [&](auto&& neighbor_data)
                {
                    using neighbor_category_t = std::decay_t<decltype(neighbor_data)>;

                    if constexpr (std::is_same_v<
                                      neighbor_category_t,
                                      typename neighbor_patch_index_variant_t::none>)
                    {
                        return;
                    }
                    else if constexpr (std::is_same_v<
                                           neighbor_category_t,
                                           typename neighbor_patch_index_variant_t::same>)
                    {
                        const auto neighbor_id = neighbor_data.id;
                        const auto opposite_d  = patch_direction_t::opposite(d);

                        const auto offsets =
                            neighbor_utils_t::compute_boundary_children(d);
                        typename neighbor_patch_index_variant_t::finer::container_t
                            fine_neighbor_ids;

                        for (auto i = 0uz; i != offsets.size(); ++i)
                        {
                            const auto offset =
                                static_cast<patch_index_t::offset_t>(offsets[i]);
                            fine_neighbor_ids[i] =
                                patch_index_t::child_of(parent_id, offset);
                        }

                        neighbor_patch_index_variant_t new_neighbor;
                        new_neighbor.data =
                            typename neighbor_patch_index_variant_t::finer{
                                fine_neighbor_ids
                            };

                        m_neighbors[m_index_map.at(neighbor_id)][opposite_d.index()] =
                            new_neighbor;
                    }
                    else if constexpr (std::is_same_v<
                                           neighbor_category_t,
                                           typename neighbor_patch_index_variant_t::
                                               finer>)
                    {
                        auto       neighbor_ids = neighbor_data.ids;
                        const auto opposite_d   = patch_direction_t::opposite(d);

                        auto offsets = neighbor_utils_t::compute_boundary_children(d);

                        for (size_t i = 0; i < neighbor_ids.size() && i < offsets.size();
                             i++)
                        {
                            auto offset =
                                static_cast<patch_index_t::offset_t>(offsets[i]);
                            auto same_neighbor_id =
                                patch_index_t::child_of(parent_id, offset);
                            neighbor_patch_index_variant_t new_neighbor;
                            new_neighbor.data =
                                typename neighbor_patch_index_variant_t::same{
                                    same_neighbor_id
                                };

                            m_neighbors[m_index_map.at(neighbor_ids[i])]
                                       [opposite_d.index()] = new_neighbor;
                        }
                    }
                    else if constexpr (std::is_same_v<
                                           neighbor_category_t,
                                           typename neighbor_patch_index_variant_t::
                                               coarser>)
                    {
                        assert(
                            false && "Coarser neighbor during refinement shouldn't happen"
                        );
                    }
                },
                neighbor.data
            );
        }
    }

    auto enforce_symmetry_after_recombining(
        patch_index_t     parent_node_id,
        patch_neighbors_t parent_neighbor
    ) noexcept -> void
    {
        for (auto d = patch_direction_t::first(); d != patch_direction_t::sentinel();
             d.advance())
        {
            const auto  opposite_d = patch_direction_t::opposite(d);
            const auto& neighbor   = parent_neighbor[d.index()];

            std::visit(
                [&](auto&& neighbor_data)
                {
                    using neighbor_category_t = std::decay_t<decltype(neighbor_data)>;

                    if constexpr (std::is_same_v<
                                      neighbor_category_t,
                                      typename neighbor_patch_index_variant_t::none>)
                    {
                        return; // No neighbor to update
                    }
                    else if constexpr (std::is_same_v<
                                           neighbor_category_t,
                                           typename neighbor_patch_index_variant_t::same>)
                    {
                        // Same-level neighbor now sees the parent instead of children
                        neighbor_patch_index_variant_t new_neighbor;
                        new_neighbor.data = typename neighbor_patch_index_variant_t::same{
                            parent_node_id
                        };
                        m_neighbors[m_index_map.at(neighbor_data.id)]
                                   [opposite_d.index()] = new_neighbor;
                    }
                    else if constexpr (std::is_same_v<
                                           neighbor_category_t,
                                           typename neighbor_patch_index_variant_t::
                                               finer>)
                    {
                        // Finer neighbors now see the parent as a coarser neighbor
                        for (size_type i = 0; i != neighbor_data.num_neighbors(); i++)
                        {
                            neighbor_patch_index_variant_t new_neighbor{
                                typename neighbor_patch_index_variant_t::coarser{
                                                                                 parent_node_id,
                                                                                 static_cast<typename neighbor_patch_index_variant_t::
                                                    fanout_t>(i) }
                            };
                            m_neighbors[m_index_map.at(neighbor_data.ids[i])]
                                       [opposite_d.index()] = new_neighbor;
                        }
                    }
                    else if constexpr (std::is_same_v<
                                           neighbor_category_t,
                                           typename neighbor_patch_index_variant_t::
                                               coarser>)
                    {
                        assert(
                            false && "Having a coarser neighbor after recombining does "
                                     "not make sense."
                        );
                    }
                },
                neighbor.data
            );
        }
    }

    auto balancing() noexcept -> void
    {
        // Refinement balancing
        for (std::size_t i = 0; i != m_to_refine.size(); ++i)
        {
            const auto node_id = m_to_refine[i];

            for (auto d = patch_direction_t::first(); d != patch_direction_t::sentinel();
                 d.advance())
            {
                const auto  neighbor_array = m_neighbors[m_index_map.at(node_id)];
                auto const& neighbor       = neighbor_array[d.index()];
                std::visit(
                    [&](auto&& neighbor_data)
                    {
                        using neighbor_category_t = std::decay_t<decltype(neighbor_data)>;
                        if constexpr (std::is_same_v<
                                          neighbor_category_t,
                                          typename neighbor_patch_index_variant_t::
                                              coarser>)
                        {
                            auto coarser_neighbor_id = neighbor_data.id;

                            auto it = std::find(
                                m_to_refine.begin(),
                                m_to_refine.end(),
                                coarser_neighbor_id
                            );
                            if (it == m_to_refine.end())
                            {
                                m_to_refine.push_back(std::move(coarser_neighbor_id));
                            }
                        }
                    },
                    neighbor.data
                );
            }
        }

        std::ranges::sort(
            m_to_refine,
            [](const auto& a, const auto& b)
            { return patch_index_t::level(a) > patch_index_t::level(b); }
        );

        std::vector<patch_index_t> blocks_to_remove{};
        for (auto i = 0uz; i != m_to_coarsen.size(); ++i)
        {
            const auto parent_id     = m_to_coarsen[i];
            bool       should_remove = false;

            for (auto d = patch_direction_t::first(); d != patch_direction_t::sentinel();
                 d.advance())
            {
                auto boundary_children = neighbor_utils_t::compute_boundary_children(d);

                // Check all boundary children for this direction
                for (auto j = 0uz; j != boundary_children.size(); ++j)
                {
                    const auto child_id = patch_index_t::child_of(
                        parent_id,
                        static_cast<typename patch_index_t::offset_t>(
                            boundary_children[j]
                        )
                    ); // Fix: variable name
                    auto const& neighbor =
                        m_neighbors[m_index_map.at(child_id)][d.index()];

                    std::visit(
                        [&](auto&& neighbor_data)
                        {
                            using neighbor_category_t =
                                std::decay_t<decltype(neighbor_data)>;
                            if constexpr (std::is_same_v<
                                              neighbor_category_t,
                                              typename neighbor_patch_index_variant_t::
                                                  finer>)
                            {
                                should_remove = true;
                            }
                            else if constexpr (std::is_same_v<
                                                   neighbor_category_t,
                                                   typename neighbor_patch_index_variant_t::
                                                       same>)
                            {
                                if (std::ranges::contains(m_to_refine, neighbor_data.id))
                                {
                                    should_remove = true;
                                }
                            }
                        },
                        neighbor.data
                    );
                    if (should_remove) break;
                }
                if (should_remove) break;
            }
            if (should_remove)
            {
                blocks_to_remove.push_back(parent_id);
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
        -> void
    {
        update_refine_flags(fn);
        apply_refine_coarsen();
        balancing();
        fragment();
        recombine();
    }

    // TODO: Use this function internally where the invariant must hold rather
    // than find_index()
    [[nodiscard]]
    auto get_linear_index_at(patch_index_t const node_id) const noexcept -> linear_index_t
    {
        const auto it = m_index_map.find(node_id);
        assert(it != nullptr && it != m_index_map.end());
        if (it == nullptr)
        {
            DEFAULT_SOURCE_LOG_FATAL("Patch index " + node_id.repr() + " not found in index map");
            utility::error_handling::assert_unreachable();
        }
        return it->second;
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

    auto append(
        patch_index_t const      node_id,
        patch_neighbors_t const& neighbor_array
    ) noexcept -> void
    {
        m_linear_index_map[m_size] = node_id;
        m_index_map[node_id]       = m_size;
        m_neighbors[m_size]        = neighbor_array;
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
        deconstructed_types_t backup_patch;
        patch_neighbors_t     backup_neighbors;

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
            backup_neighbors     = m_neighbors[i];

            // Backup entire patch
            [this, i, &backup_patch]<std::size_t... I>(std::index_sequence<I...>)
            {
                ((void)(std::get<I>(backup_patch) = std::get<I>(m_data_buffers)[i]), ...);
            }(std::make_index_sequence<std::tuple_size_v<deconstructed_buffers_t>>{});

            auto dst = i;
            do
            {
                m_linear_index_map[dst]     = m_linear_index_map[src];
                m_refine_status_buffer[dst] = m_refine_status_buffer[src];
                m_neighbors[dst]            = m_neighbors[src];

                // Copy entire patches
                std::apply(
                    [dst, src](auto&... b) { ((void)(b[dst] = b[src]), ...); },
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
            m_neighbors[dst]            = backup_neighbors;

            // Restore backed up patch
            [this, dst, &backup_patch]<std::size_t... I>(std::index_sequence<I...>)
            {
                ((void)(std::get<I>(m_data_buffers)[dst] = std::get<I>(backup_patch)),
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
    auto gather_node(linear_index_t const i) const noexcept -> map_type
    {
        DEFAULT_SOURCE_LOG_TRACE("Gather node");
        return std::apply(
            [i](auto&&... args)
            { return map_type(std::forward<decltype(args)>(args)[i]...); },
            m_data_buffers
        );
    }

    auto scatter_node(map_type const& v, const linear_index_t i) const noexcept -> void
    {
        DEFAULT_SOURCE_LOG_TRACE("Scatter node");
        [this, &v, i]<std::size_t... I>(std::index_sequence<I...>)
        {
            ((void)(std::get<I>(m_data_buffers)[i] = std::get<I>(v.data_tuple()).value),
             ...);
        }(std::make_index_sequence<std::tuple_size_v<deconstructed_buffers_t>>{});
    }

    auto restrict_patches(
        linear_index_t const start_from,
        linear_index_t const to
    ) noexcept -> void
    {
        DEFAULT_SOURCE_LOG_TRACE("Patch restriction");
        static constexpr auto patch_size = patch_layout_t::flat_size();

        // TODO: remove
        std::apply(
            [to](auto&... b)
            {
                for (linear_index_t k = 0; k != patch_size; k++)
                {
                    if (utils::patches::is_halo_cell<patch_layout_t>(k))
                    {
                        continue;
                    }
                    ((b[to][k] = static_cast<unwrap_value_t<decltype(b)>>(0)), ...);
                }
            },
            m_data_buffers
        );

        // TODO: The solver needs to decide how to do the interpolation
        for (size_type patch_idx = 0; patch_idx != s_nd_fanout; ++patch_idx)
        {
            const auto child_patch_index = start_from + patch_idx;
            for (linear_index_t linear_idx = 0; linear_idx != patch_size; ++linear_idx)
            {
                if (utils::patches::is_halo_cell<patch_layout_t>(linear_idx))
                {
                    continue;
                }
                const auto to_linear_idx =
                    s_fragmentation_patch_maps[patch_idx][linear_idx];

                std::apply(
                    [to, to_linear_idx, child_patch_index, linear_idx](auto&... b)
                    {
                        ((void)(b[to][to_linear_idx] += b[child_patch_index][linear_idx]),
                         ...);
                    },
                    m_data_buffers
                );
            }
        }
        std::apply(
            [to](auto&... b)
            {
                for (linear_index_t k = 0; k != patch_size; k++)
                {
                    if (utils::patches::is_halo_cell<patch_layout_t>(k))
                    {
                        continue;
                    }
                    ((b[to][k] /= static_cast<unwrap_value_t<decltype(b)>>(s_nd_fanout)),
                     ...);
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
        DEFAULT_SOURCE_LOG_TRACE("Patch interpolation");
        for (size_type patch_idx = 0; patch_idx != s_nd_fanout; ++patch_idx)
        {
            const auto child_patch_index = start_to + patch_idx;
            for (linear_index_t linear_idx = 0; linear_idx != patch_layout_t::flat_size();
                 ++linear_idx)
            {
                if (utils::patches::is_halo_cell<patch_layout_t>(linear_idx))
                {
                    continue;
                }
                const auto from_linear_idx =
                    s_fragmentation_patch_maps[patch_idx][linear_idx];
                std::apply(
                    [child_patch_index, from, from_linear_idx, linear_idx](auto&... b)
                    {
                        ((void)(b[child_patch_index][linear_idx] =
                                    b[from][from_linear_idx]),
                         ...);
                    },
                    m_data_buffers
                );
            }
        }
    }

    constexpr auto halo_exchange_update() noexcept -> void
    {
        DEFAULT_SOURCE_LOG_TRACE("Performing halo exchange");
        for (linear_index_t i = 0; i != m_size; ++i)
        {
            utils::patches::halo_apply<halo_exchange_operator_impl_t, patch_direction_t>(
                *this, i
            );
        }
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

    auto compact() noexcept -> void
    {
        DEFAULT_SOURCE_LOG_TRACE("Comacting tree");
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
        std::swap(m_neighbors[i], m_neighbors[j]);

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
                    DEFAULT_SOURCE_LOG_ERROR("index map is not correct");
                    return false;
                }
            }
            return true;
        }
        DEFAULT_SOURCE_LOG_ERROR("linear index is not sorted");
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
        DEFAULT_SOURCE_LOG_ERROR("linear index is not sorted");
    }
#endif

private:
    index_map_t                m_index_map;
    deconstructed_buffers_t    m_data_buffers;
    linear_index_map_t         m_linear_index_map;
    linear_index_array_t       m_reorder_buffer;
    flat_refine_status_array_t m_refine_status_buffer;
    neighbor_buffer_t          m_neighbors;
    size_type                  m_size;
    std::vector<patch_index_t> m_to_refine;
    std::vector<patch_index_t> m_to_coarsen;
};

} // namespace amr::ndt::tree

#endif // AMR_INCLUDED_NDTREE
