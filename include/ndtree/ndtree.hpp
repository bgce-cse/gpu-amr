#ifndef AMR_INCLUDED_NDTREE
#define AMR_INCLUDED_NDTREE

#include "config/definitions.hpp"
#include "intergrid_operator.hpp"
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
#include <execution>
#include <numeric>
#include <ranges>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#define AMR_NDTREE_ENABLE_CHECKS

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
    using map_type          = T;
    using size_type         = std::size_t;
    using patch_index_t     = Patch_Index;
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
    static constexpr size_type s_rank       = patch_layout_t::rank();

    static_assert(s_1d_fanout > 1);
    static_assert(s_nd_fanout > 1);
    static_assert(
        utils::patches::multiples_of(patch_layout_t::data_layout_t::sizes(), s_1d_fanout),
        "All patch dimensions must be multiples of the fanout"
    );
    static_assert(
        std::ranges::all_of(
            patch_layout_t::data_layout_t::sizes(),
            [](auto const& e)
            { return e >= static_cast<decltype(e)>(s_halo_width * s_1d_fanout); }
        ),
        "The halo must not span more than one finer neighbor cell"
    );
    static_assert(std::is_same_v<
                  typename neighbor_patch_index_variant_t::index_t,
                  typename neighbor_utils_t::index_t>);

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
    static constexpr auto fragmentation_patch_maps(
        const size_type      patch_idx,
        const linear_index_t linear_idx
    ) noexcept -> typename patch_layout_t::index_t
    {
#if defined(__clang__) || defined(__NVCOMPILER)
        static const auto s_fragmentation_patch_maps = amr::ndt::utils::patches::
            fragmentation_patch_maps<patch_layout_t, s_1d_fanout>();
#else
        static constexpr auto s_fragmentation_patch_maps = amr::ndt::utils::patches::
            fragmentation_patch_maps<patch_layout_t, s_1d_fanout>();
#endif
        return s_fragmentation_patch_maps[patch_idx][linear_idx];
    }

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

private:
    [[nodiscard]]
    auto alloc_slot() noexcept -> linear_index_t
    {
        assert(
            m_free_top < m_capacity && "ndtree: out of patch slots — increase capacity"
        );
        const auto ret             = m_free_slots[m_free_top];
        m_free_slots[m_free_top++] = m_capacity;
        return ret;
    }

    auto free_slot(linear_index_t const slot) noexcept -> void
    {
        assert(m_free_top > 0 && "ndtree: no slots are currenlty allocated");
        assert(slot < m_capacity && "ndtree: free_slot out of range");
        assert(m_slot_active[slot] && "ndtree: double-free of slot");
        m_slot_active[slot]        = false;
        m_free_slots[--m_free_top] = slot;
    }

    auto write_slot_metadata(
        linear_index_t const     slot,
        patch_index_t const      node_id,
        patch_neighbors_t const& neighbors
    ) noexcept -> void
    {
        DEFAULT_SOURCE_LOG_TRACE("Trying to allocate at slot: {}\n", slot);
        assert(slot < m_capacity && "ndtree: write_slot_metadata out of range");
        assert(!m_slot_active[slot] && "ndtree: write_slot_metadata on active slot");
        m_linear_index_map[slot]     = node_id;
        m_neighbors[slot]            = neighbors;
        m_refine_status_buffer[slot] = RefinementStatus::Stable;
        m_slot_active[slot]          = true;
        m_index_map[node_id]         = slot;
        m_active_slots[m_size++]     = slot;
    }

    auto remove_slot_metadata(linear_index_t const slot) noexcept -> void
    {
        for (linear_index_t i = 0; i != m_size; ++i)
        {
            if (m_active_slots[i] == slot)
            {
                m_active_slots[i] = m_active_slots[--m_size];
                return;
            }
        }
        assert(false && "ndtree: slot not found in m_active_slots");
    }

public:
    ndtree(size_type const capacity) noexcept
        : m_index_map{} // , m_slot_active(m_capacity, false)
        , m_size{}
        , m_capacity{ capacity }
        , m_free_top{}
    {
        std::apply(
            [capacity](auto&... b)
            {
                ((void)(b = static_cast<pointer_t<value_t<decltype(b)>>>(
                            std::malloc(capacity * sizeof(value_t<decltype(b)>))
                        )),
                 ...);
            },
            m_data_buffers
        );

        m_linear_index_map = static_cast<pointer_t<patch_index_t>>(
            std::malloc(capacity * sizeof(patch_index_t))
        );
        m_refine_status_buffer = static_cast<pointer_t<refine_status_t>>(
            std::malloc(capacity * sizeof(refine_status_t))
        );
        m_neighbors = static_cast<pointer_t<patch_neighbors_t>>(
            std::malloc(capacity * sizeof(patch_neighbors_t))
        );
        m_slot_active =
            static_cast<pointer_t<bool>>(std::malloc(capacity * sizeof(bool)));
        m_active_slots = static_cast<pointer_t<linear_index_t>>(
            std::malloc(capacity * sizeof(linear_index_t))
        );
        m_free_slots = static_cast<pointer_t<linear_index_t>>(
            std::malloc(capacity * sizeof(linear_index_t))
        );

        std::iota(m_free_slots, m_free_slots + capacity, linear_index_t{ 0 });
        std::fill(m_slot_active, m_slot_active + capacity, false);

        patch_neighbors_t root_neighbors;
        for (auto d = patch_direction_t::first(); d != patch_direction_t::sentinel();
             d.advance())
        {
            neighbor_patch_index_variant_t periodic_neighbor;
            periodic_neighbor.data =
                typename neighbor_patch_index_variant_t::same{ patch_index_t::root() };
            root_neighbors[d.index()] = periodic_neighbor;
        }
        const auto root_slot = alloc_slot();
        write_slot_metadata(root_slot, patch_index_t::root(), root_neighbors);
    }

    ~ndtree() noexcept
    {
        std::apply([](auto&... b) { ((void)std::free(b), ...); }, m_data_buffers);
        std::free(m_linear_index_map);
        std::free(m_refine_status_buffer);
        std::free(m_neighbors);
        std::free(m_slot_active);
        std::free(m_active_slots);
        std::free(m_free_slots);
    }

public:
    [[nodiscard]]
    auto size() const noexcept -> size_type
    {
        return m_size;
    }

    [[nodiscard]]
    auto capacity() const noexcept -> size_type
    {
        return m_capacity;
    }

    [[nodiscard]]
    auto active_slots() const noexcept -> linear_index_t const*
    {
        return m_active_slots;
    }

    [[nodiscard]]
    auto get_active_slot_at(linear_index_t const i) const noexcept -> linear_index_t
    {
        return m_active_slots[i];
    }

    template <concepts::MapType Map_Type>
    [[nodiscard, gnu::always_inline, gnu::flatten]]
    auto get_patch(linear_index_t const slot) noexcept -> patch_t<Map_Type>&
    {
        return const_cast<patch_t<Map_Type>&>(
            std::as_const(*this).template get_patch<Map_Type>(slot)
        );
    }

    template <concepts::MapType Map_Type>
    [[nodiscard, gnu::always_inline, gnu::flatten]]
    auto get_patch(linear_index_t const slot) const noexcept -> patch_t<Map_Type> const&
    {
        assert(slot < m_capacity && "get_patch: slot out of range");
        assert(m_slot_active[slot] && "get_patch: slot not active");
        return std::get<Map_Type::index()>(m_data_buffers)[slot];
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
        const auto slot = m_index_map.at(patch_idx);
        assert(slot < m_capacity && "get_patch: slot out of range");
        assert(m_slot_active[slot] && "get_patch: slot not active");
        return std::get<Map_Type::index()>(m_data_buffers)[slot];
    }

    [[nodiscard, gnu::always_inline, gnu::flatten]]
    auto get_neighbor_at(
        linear_index_t const     slot,
        patch_direction_t const& d
    ) const noexcept -> neighbor_patch_index_variant_t
    {
        assert(slot < m_capacity && m_slot_active[slot]);
        return m_neighbors[slot][d.index()];
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
                { return ret_t{}; },
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
                        typename ret_t::coarser{ get_linear_index_at(n.id),
                                                n.contact_quadrant }
                    };
                } },
            neighbor.data
        );
    }

    [[nodiscard]]
    auto get_refine_status(linear_index_t const slot) const noexcept -> refine_status_t
    {
        assert(slot < m_capacity && m_slot_active[slot]);
        return m_refine_status_buffer[slot];
    }

    [[nodiscard]]
    auto get_linear_index_at(patch_index_t const node_id) const noexcept -> linear_index_t
    {
        assert(
            m_index_map.find(node_id) != m_index_map.end() &&
            "get_linear_index_at: patch_index not found"
        );
        return m_index_map.at(node_id);
    }

    [[nodiscard]]
    auto get_node_index_at(linear_index_t const slot) const noexcept -> patch_index_t
    {
        assert(slot < m_capacity && m_slot_active[slot]);
        return m_linear_index_map[slot];
    }

    [[nodiscard]]
    auto gather_node(linear_index_t const slot) const noexcept -> map_type
    {
        DEFAULT_SOURCE_LOG_TRACE("Gather node");
        assert(slot < m_capacity && m_slot_active[slot]);
        return std::apply(
            [slot](auto&&... args)
            { return map_type(std::forward<decltype(args)>(args)[slot]...); },
            m_data_buffers
        );
    }

    auto scatter_node(map_type const& v, linear_index_t const slot) noexcept -> void
    {
        DEFAULT_SOURCE_LOG_TRACE("Scatter node");
        assert(slot < m_capacity && m_slot_active[slot]);
        [this, &v, slot]<std::size_t... I>(std::index_sequence<I...>)
        {
            ((void)(std::get<I>(m_data_buffers)[slot] =
                        std::get<I>(v.data_tuple()).value),
             ...);
        }(std::make_index_sequence<std::tuple_size_v<deconstructed_buffers_t>>{});
    }

public:
    template <typename Fn>
    auto reconstruct_tree(Fn&& fn) noexcept(noexcept(fn(std::declval<patch_index_t&>())))
        -> void
    {
        update_refine_flags(std::forward<Fn>(fn));
        apply_refine_coarsen();
        balancing();
        fragment();
        recombine();
        compact();
        sort_active_slots();
    }

    template <typename Fn>
    auto update_refine_flags(Fn&& fn) noexcept(
        noexcept(fn(std::declval<patch_index_t&>()))
    ) -> void
    {
        for (linear_index_t i = 0; i != m_size; ++i)
        {
            const auto slot              = m_active_slots[i];
            m_refine_status_buffer[slot] = fn(m_linear_index_map[slot]);
        }
    }

    auto apply_refine_coarsen() noexcept -> void
    {
        m_to_refine.clear();
        m_to_coarsen.clear();
        std::vector<patch_index_t> parent_patch_idx;

        for (linear_index_t i = 0; i != m_size; ++i)
        {
            const auto slot    = m_active_slots[i];
            const auto node_id = m_linear_index_map[slot];
            if (node_id.id() == 0) continue;
            parent_patch_idx.push_back(patch_index_t::parent_of(node_id));
        }

        std::ranges::sort(parent_patch_idx, std::less{});
        parent_patch_idx.erase(
            std::unique(parent_patch_idx.begin(), parent_patch_idx.end()),
            parent_patch_idx.end()
        );

        for (linear_index_t i = 0; i != m_size; ++i)
        {
            const auto slot = m_active_slots[i];
            if (is_refine_elegible(slot)) m_to_refine.push_back(m_linear_index_map[slot]);
        }
        for (auto const& parent_id : parent_patch_idx)
        {
            if (is_coarsen_elegible(parent_id)) m_to_coarsen.push_back(parent_id);
        }
    }

    auto fragment(patch_index_t const node_id) -> void
    {
        DEFAULT_SOURCE_LOG_TRACE("Fragmenting node " + node_id.repr());

        const auto parent_slot = m_index_map.at(node_id);
        assert(m_slot_active[parent_slot]);

        std::array<linear_index_t, s_nd_fanout> child_slots;
        for (size_type i = 0; i != s_nd_fanout; ++i)
        {
            const auto child_id = patch_index_t::child_of(
                node_id, static_cast<typename patch_index_t::offset_t>(i)
            );
            assert(!find_index(child_id).has_value());

            const auto child_slot   = alloc_slot();
            const auto neighbor_arr = neighbor_utils_t::compute_child_neighbors(
                node_id, m_neighbors[parent_slot], i
            );

            write_slot_metadata(child_slot, child_id, neighbor_arr);
            child_slots[i] = child_slot;
        }

        interpolate_patch(parent_slot, child_slots);
        enforce_symmetry_after_refinement(node_id);

        remove_slot_metadata(parent_slot);
        m_index_map.erase(node_id);
        free_slot(parent_slot);

#ifdef AMR_NDTREE_ENABLE_CHECKS
        check_index_map();
#endif
    }

    auto recombine(patch_index_t const parent_node_id) -> void
    {
        using offset_t = typename patch_index_t::offset_t;
        assert(!find_index(parent_node_id).has_value());

        std::array<linear_index_t, s_nd_fanout>    child_slots;
        std::array<patch_neighbors_t, s_nd_fanout> child_neighbor_arrays{};
        for (size_type i = 0; i != s_nd_fanout; ++i)
        {
            const auto child_id =
                patch_index_t::child_of(parent_node_id, static_cast<offset_t>(i));
            const auto child_slot    = m_index_map.at(child_id);
            child_slots[i]           = child_slot;
            child_neighbor_arrays[i] = m_neighbors[child_slot];
        }

        const auto parent_slot = alloc_slot();
        const auto neighbor_array =
            neighbor_utils_t::compute_parent_neighbors(child_neighbor_arrays);
        write_slot_metadata(parent_slot, parent_node_id, neighbor_array);

        restrict_patches(child_slots, parent_slot);

        for (size_type i = 0; i != s_nd_fanout; ++i)
        {
            const auto child_id =
                patch_index_t::child_of(parent_node_id, static_cast<offset_t>(i));
            remove_slot_metadata(child_slots[i]);
            m_index_map.erase(child_id);
            free_slot(child_slots[i]);
        }

        enforce_symmetry_after_recombining(parent_node_id, neighbor_array);
    }

    auto fragment() -> void
    {
        for (auto i = m_to_refine.size(); i > 0; --i)
            fragment(m_to_refine[i - 1]);
    }

    auto recombine() -> void
    {
        for (const auto& node_id : m_to_coarsen)
            recombine(node_id);
    }

    auto enforce_symmetry_after_refinement(patch_index_t const parent_id) -> void
    {
        for (auto d = patch_direction_t::first(); d != patch_direction_t::sentinel();
             d.advance())
        {
            const auto opposite_d        = patch_direction_t::opposite(d);
            const auto boundary_children = neighbor_utils_t::compute_boundary_children(d);

            auto const& parent_neighbor =
                m_neighbors[m_index_map.at(parent_id)][d.index()];

            std::visit(
                [&](auto const& neighbor_data)
                {
                    using neighbor_category_t = std::decay_t<decltype(neighbor_data)>;

                    if constexpr (std::is_same_v<
                                      neighbor_category_t,
                                      typename neighbor_patch_index_variant_t::same>)
                    {
                        typename neighbor_patch_index_variant_t::finer::container_t
                            fine_ids{};
                        for (size_type i = 0; i != boundary_children.size(); ++i)
                        {
                            const auto child_id = patch_index_t::child_of(
                                parent_id,
                                static_cast<typename patch_index_t::offset_t>(
                                    boundary_children[i]
                                )
                            );
                            fine_ids[i] = child_id;
                        }
                        neighbor_patch_index_variant_t new_neighbor;
                        new_neighbor.data =
                            typename neighbor_patch_index_variant_t::finer{ fine_ids };
                        m_neighbors[m_index_map.at(neighbor_data.id)]
                                   [opposite_d.index()] = new_neighbor;
                    }
                    else if constexpr (std::is_same_v<
                                           neighbor_category_t,
                                           typename neighbor_patch_index_variant_t::
                                               finer>)
                    {
                        for (size_type i = 0; i != boundary_children.size(); ++i)
                        {
                            const auto child_id = patch_index_t::child_of(
                                parent_id,
                                static_cast<typename patch_index_t::offset_t>(
                                    boundary_children[i]
                                )
                            );
                            neighbor_patch_index_variant_t new_neighbor;
                            new_neighbor.data =
                                typename neighbor_patch_index_variant_t::same{ child_id };
                            m_neighbors[m_index_map.at(neighbor_data.ids[i])]
                                       [opposite_d.index()] = new_neighbor;
                        }
                    }
                },
                parent_neighbor.data
            );
        }
    }

    auto enforce_symmetry_after_recombining(
        patch_index_t const      parent_node_id,
        patch_neighbors_t const& parent_neighbor_array
    ) -> void
    {
        for (auto d = patch_direction_t::first(); d != patch_direction_t::sentinel();
             d.advance())
        {
            const auto  opposite_d      = patch_direction_t::opposite(d);
            auto const& parent_neighbor = parent_neighbor_array[d.index()];

            std::visit(
                [&](auto const& neighbor_data)
                {
                    using neighbor_category_t = std::decay_t<decltype(neighbor_data)>;

                    if constexpr (std::is_same_v<
                                      neighbor_category_t,
                                      typename neighbor_patch_index_variant_t::same>)
                    {
                        const auto it = m_index_map.find(neighbor_data.id);
                        if (it == m_index_map.end()) return;
                        neighbor_patch_index_variant_t new_neighbor;
                        new_neighbor.data = typename neighbor_patch_index_variant_t::same{
                            parent_node_id
                        };
                        m_neighbors[it->second][opposite_d.index()] = new_neighbor;
                    }
                    else if constexpr (std::is_same_v<
                                           neighbor_category_t,
                                           typename neighbor_patch_index_variant_t::
                                               finer>)
                    {
                        for (size_type i = 0; i != neighbor_data.ids.size(); ++i)
                        {
                            const auto it = m_index_map.find(neighbor_data.ids[i]);
                            if (it == m_index_map.end()) continue;
                            const auto contact_q =
                                neighbor_utils_t::compute_contact_quadrant(
                                    static_cast<
                                        typename neighbor_patch_index_variant_t::index_t>(
                                        i
                                    ),
                                    opposite_d
                                );
                            neighbor_patch_index_variant_t new_neighbor;
                            new_neighbor.data =
                                typename neighbor_patch_index_variant_t::coarser{
                                    parent_node_id, contact_q
                                };
                            m_neighbors[it->second][opposite_d.index()] = new_neighbor;
                        }
                    }
                    else if constexpr (std::is_same_v<
                                           neighbor_category_t,
                                           typename neighbor_patch_index_variant_t::
                                               coarser>)
                    {
                        CONTRACTS_CHECK(
                            false && "Having a coarser neighbor after recombining does "
                                     "not make sense."
                        );
                    }
                },
                parent_neighbor.data
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

                for (auto j = 0uz; j != boundary_children.size(); ++j)
                {
                    const auto child_id = patch_index_t::child_of(
                        parent_id,
                        static_cast<typename patch_index_t::offset_t>(
                            boundary_children[j]
                        )
                    );
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

    constexpr auto halo_exchange_update() noexcept -> void
    {
        DEFAULT_SOURCE_LOG_TRACE("Performing halo exchange");
        for (linear_index_t i = 0; i != m_size; ++i)
        {
            const auto slot = m_active_slots[i];
            utils::patches::halo_apply<halo_exchange_operator_impl_t, patch_direction_t>(
                *this, slot
            );
        }
    }

    auto restrict_patches(
        std::array<linear_index_t, s_nd_fanout> const& child_slots,
        linear_index_t const                           parent_slot
    ) noexcept -> void
    {
        DEFAULT_SOURCE_LOG_TRACE("Patch restriction");
        using hypercube_t = amr::containers::utils::types::tensor::hypercube_t<
            typename patch_layout_t::padded_layout_t::index_t,
            s_1d_fanout,
            s_rank>;

        amr::containers::manipulators::shaped_for<
            typename patch_layout_t::interior_iteration_control_t>(
            [this, parent_slot, &child_slots](auto const& idxs)
            {
                auto fine_patch_idxs = idxs;
                auto base_fine_idxs  = idxs;

                for (size_type d = 0; d != s_rank; ++d)
                {
                    const auto section_size =
                        patch_layout_t::data_layout_t::size(d) / s_1d_fanout;
                    fine_patch_idxs[d] =
                        (fine_patch_idxs[d] - s_halo_width) / section_size;
                    base_fine_idxs[d] =
                        ((base_fine_idxs[d] - s_halo_width) % section_size) *
                            s_1d_fanout +
                        s_halo_width;
                }

                const auto fine_patch_idx =
                    hypercube_t::layout_t::linear_index(fine_patch_idxs);
                const auto child_slot = child_slots[fine_patch_idx];

                const auto base_fine_idx =
                    patch_layout_t::padded_layout_t::linear_index(base_fine_idxs);
                const auto fine_linear_idxs =
                    utils::patches::detail::hypercube_offset<patch_layout_t, 2>(
                        base_fine_idx
                    );

                std::apply(
                    [parent_slot, &idxs, child_slot, fine_linear_idxs](auto&... b)
                    {
                        ((b[parent_slot][idxs] =
                              amr::ndt::intergrid_operator::linear_interpolator<
                                  patch_layout_t>::restriction(
                                  b[child_slot], fine_linear_idxs
                              )),
                         ...);
                    },
                    m_data_buffers
                );
            }
        );
    }

    auto interpolate_patch(
        linear_index_t const                           parent_slot,
        std::array<linear_index_t, s_nd_fanout> const& child_slots
    ) noexcept -> void
    {
        DEFAULT_SOURCE_LOG_TRACE("Patch interpolation");
        for (size_type patch_idx = 0; patch_idx != s_nd_fanout; ++patch_idx)
        {
            const auto child_slot = child_slots[patch_idx];
            for (linear_index_t linear_idx = 0; linear_idx != patch_layout_t::flat_size();
                 ++linear_idx)
            {
                if (utils::patches::is_halo_cell<patch_layout_t>(linear_idx)) continue;
                const auto from_linear_idx =
                    fragmentation_patch_maps(patch_idx, linear_idx);
                std::apply(
                    [child_slot, parent_slot, from_linear_idx, linear_idx](auto&... b)
                    {
                        ((void)(b[child_slot][linear_idx] =
                                    b[parent_slot][from_linear_idx]),
                         ...);
                    },
                    m_data_buffers
                );
            }
        }
    }

private:
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

    [[nodiscard]]
    auto is_refine_elegible(linear_index_t const slot) const noexcept -> bool
    {
        assert(slot < m_capacity && m_slot_active[slot]);
        return (m_refine_status_buffer[slot] == RefinementStatus::Refine) &&
               (patch_index_t::level(m_linear_index_map[slot]) <
                patch_index_t::max_depth());
    }

    [[nodiscard]]
    auto is_coarsen_elegible(patch_index_t const parent_id) const noexcept -> bool
    {
        for (size_type i = 0; i != s_nd_fanout; ++i)
        {
            const auto child = patch_index_t::child_of(
                parent_id, static_cast<typename patch_index_t::offset_t>(i)
            );
            const auto it = find_index(child);
            if (!it.has_value()) return false;
            const auto slot = it.value()->second;
            if (m_refine_status_buffer[slot] != RefinementStatus::Coarsen) return false;
        }
        return true;
    }

    auto compact() noexcept -> void
    {
        DEFAULT_SOURCE_LOG_TRACE("Compacting tree");
        size_type write = 0;
        for (linear_index_t i = 0; i != m_size; ++i)
        {
            const auto read = m_active_slots[i];
            if (read < m_size)
            {
                continue;
            }
            else
            {
                for (; write != m_size;)
                {
                    const auto w = write++;
                    if (m_slot_active[w])
                    {
                        continue;
                    }
                    else
                    {
                        block_buffer_swap(w, read);
                        m_active_slots[i]                  = w;
                        m_index_map[m_linear_index_map[w]] = w;
                        auto it                            = std::find(
                            m_free_slots + m_free_top, m_free_slots + m_capacity, w
                        );
                        assert(it != m_free_slots + m_capacity);
                        *it = read;
                        break;
                    }
                }
            }
        }

        assert(m_index_map.size() == m_size);
        for (linear_index_t i = 0; i != m_size; ++i)
        {
            [[maybe_unused]]
            const auto slot = m_active_slots[i];
            assert(slot < m_size);
            assert(m_slot_active[slot]);
            assert(m_index_map[m_linear_index_map[slot]] == slot);
        }
        for (linear_index_t i = 0; i != m_size; ++i)
        {
            assert(m_slot_active[i]);
        }
        for (linear_index_t i = m_size; i != m_capacity; ++i)
        {
            assert(!m_slot_active[i]);
        }
        check_index_map();
    }

    auto sort_active_slots() noexcept -> void
    {
        std::sort(
            m_active_slots,
            m_active_slots + m_size,
            [this](linear_index_t const lhs, linear_index_t const rhs)
            { return m_linear_index_map[lhs] < m_linear_index_map[rhs]; }
        );
    }

    [[gnu::always_inline, gnu::flatten]]
    auto block_buffer_swap(linear_index_t const i, linear_index_t const j) noexcept
        -> void

    {
        CONTRACTS_CHECK(i < m_capacity);
        CONTRACTS_CHECK(j < m_capacity);
        CONTRACTS_CHECK(i < m_size || j < m_size);
        if (i == j)
        {
            return;
        }

        CONTRACTS_CHECK(m_linear_index_map[i] != m_linear_index_map[j]);
        std::swap(m_linear_index_map[i], m_linear_index_map[j]);
        std::swap(m_refine_status_buffer[i], m_refine_status_buffer[j]);
        std::swap(m_neighbors[i], m_neighbors[j]);
        std::swap(m_slot_active[i], m_slot_active[j]);
        std::apply(
            [i, j](auto&... b) { ((void)std::swap(b[i], b[j]), ...); }, m_data_buffers
        );
    }

#ifdef AMR_NDTREE_ENABLE_CHECKS
    auto check_index_map() const noexcept -> void
    {
        for (const auto& [node_idx, slot] : m_index_map)
        {
            if (m_linear_index_map[slot] != node_idx)
            {
                DEFAULT_SOURCE_LOG_ERROR("morton index map is incorrect");
                assert(false);
            }
            if (!m_slot_active[slot])
            {
                DEFAULT_SOURCE_LOG_ERROR("index_map points to inactive slot");
                assert(false);
            }
        }
        if (m_index_map.size() != m_size)
        {
            DEFAULT_SOURCE_LOG_ERROR("index_map size != m_size");
            assert(false);
        }
    }
#endif

private:
    index_map_t                m_index_map;            // patch_index → slot
    deconstructed_buffers_t    m_data_buffers;         // [field][slot] — NEVER moves
    linear_index_map_t         m_linear_index_map;     // [slot] → patch_index
    linear_index_array_t       m_active_slots;         // [iter_pos] → slot (unordered)
    linear_index_array_t       m_free_slots;           // flat free-list stack
    pointer_t<bool>            m_slot_active;          // [slot] → is live
    flat_refine_status_array_t m_refine_status_buffer; // [slot] → RefinementStatus
    neighbor_buffer_t          m_neighbors;            // [slot] → patch_neighbors_t
    size_type                  m_size;                 // active patch count
    size_type                  m_capacity;             // total allocated slots
    size_type                  m_free_top;             // stack pointer into m_free_slots
    std::vector<patch_index_t> m_to_refine;
    std::vector<patch_index_t> m_to_coarsen;
};

} // namespace amr::ndt::tree

#endif // AMR_INCLUDED_NDTREE
