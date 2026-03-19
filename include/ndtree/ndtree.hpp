#ifndef AMR_INCLUDED_NDTREE
#define AMR_INCLUDED_NDTREE

#ifdef AMR_ENABLE_CUDA_AMR
#    include "cuda/device_buffer.hpp"
#    include "cuda/halo_exchange.hpp"
#    include "cuda/intergrid_transfer.hpp"
#    include "cuda/permutation.hpp"
#endif
#include "config/definitions.hpp"
#include "ndconcepts.hpp"
#include "ndtype_traits.hpp"
#include "ndutils.hpp"
#include "neighbor.hpp"
#include "patch.hpp"
#include "patch_utils.hpp"
#include "utility/compile_time_utility.hpp"
#include "utility/error_handling.hpp"
#include "utility/logging.hpp"
#ifdef AMR_DG_COARSEN_DEBUG
#    include <iostream>
#endif
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <execution>
#include <numeric>
#include <ranges>
#include <span>
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
    concepts::PatchLayout         Patch_Layout,
    concepts::IntergridOperator   Intergrid_Operator>
class ndtree
{
public:
    using map_type             = T;
    using size_type            = std::size_t;
    using patch_index_t        = Patch_Index;
    using intergrid_operator_t = Intergrid_Operator;
    using linear_index_t       = size_type;
    using patch_layout_t       = Patch_Layout;
    using neighbor_utils_t     = neighbors::neighbor_utils<patch_index_t, patch_layout_t>;
    using patch_direction_t    = typename neighbor_utils_t::direction_t;
    template <typename Identifier_Type>
    using neighbor_variant_base_t =
        typename neighbor_utils_t::template neighbor_variant_base_t<Identifier_Type>;
    using neighbor_patch_index_variant_t  = neighbor_variant_base_t<patch_index_t>;
    using neighbor_linear_index_variant_t = neighbor_variant_base_t<linear_index_t>;
    using patch_neighbors_t               = typename neighbor_utils_t::patch_neighbors_t;
    using halo_exchange_operator_impl_t   = utils::patches::
        halo_exchange_impl_t<patch_index_t, patch_layout_t, intergrid_operator_t>;
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
            [](auto const& e) { return e >= s_halo_width * s_1d_fanout; }
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

    using projection_hypercube_t =
        typename halo_exchange_operator_impl_t::projection_hypercube_t;

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


    //safety check bc gpu tree version onlyworks with doubles stored in the patche rn
#ifdef AMR_ENABLE_CUDA_AMR
    template <typename Map_Type>
    static constexpr bool s_cuda_scalar_halo_map =
        std::is_same_v<typename Map_Type::type, double>;

    template <std::size_t... I>
    static consteval auto cuda_scalar_halo_compatible_impl(std::index_sequence<I...>) -> bool
    {
        return (... && s_cuda_scalar_halo_map<
            std::tuple_element_t<I, deconstructed_raw_map_types_t>>);
    }

    static constexpr bool s_cuda_scalar_patch_compatible =
        cuda_scalar_halo_compatible_impl(
            std::make_index_sequence<std::tuple_size_v<deconstructed_raw_map_types_t>>{}
        ) &&
        (s_rank <= 3) && (s_1d_fanout == 2);

    static constexpr bool s_cuda_scalar_halo_compatible = s_cuda_scalar_patch_compatible;

    using device_transfer_task_t = amr::cuda::intergrid_transfer_task_metadata;
#endif

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

public:
    struct current_buffer_t
    {
        template <typename U>
        constexpr auto operator==(U) const noexcept -> bool
        {
            return std::is_same_v<current_buffer_t, U>;
        };
    };

    static constexpr current_buffer_t current_buffer{};

    struct next_buffer_t
    {
        template <typename U>
        constexpr auto operator==(U) const noexcept -> bool
        {
            return std::is_same_v<next_buffer_t, U>;
        };
    };

    static constexpr next_buffer_t next_buffer{};

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
        std::apply(
            [size](auto&... b)
            {
                ((void)(b = (pointer_t<value_t<decltype(b)>>)
                            std::malloc(size * sizeof(value_t<decltype(b)>))),
                 ...);
            },
            m_next_buffers
        );
#ifdef AMR_ENABLE_CUDA_AMR
        std::apply(
            [size](auto&... b)
            {
                ((void)(b = static_cast<pointer_t<value_t<decltype(b)>>>(
                            amr::cuda::device_malloc(size * sizeof(value_t<decltype(b)>))
                        )),
                 ...);
            },
            m_device_data_buffers
        );
        std::apply(
            [size](auto&... b)
            {
                ((void)(b = static_cast<pointer_t<value_t<decltype(b)>>>(
                            amr::cuda::device_malloc(size * sizeof(value_t<decltype(b)>))
                        )),
                 ...);
            },
            m_device_next_buffers
        );
#endif
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
        std::apply([](auto&... b) { ((void)std::free(b), ...); }, m_next_buffers);
        std::apply([](auto&... b) { ((void)std::free(b), ...); }, m_data_buffers);
#ifdef AMR_ENABLE_CUDA_AMR
        std::apply(
            [](auto&... b) { ((void)amr::cuda::device_free(static_cast<void*>(b)), ...); },
            m_device_next_buffers
        );
        std::apply(
            [](auto&... b) { ((void)amr::cuda::device_free(static_cast<void*>(b)), ...); },
            m_device_data_buffers
        );
#endif
    }

public:
    [[nodiscard]]
    auto size() const noexcept -> size_type
    {
        return m_size;
    }

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
        CONTRACTS_CHECK(linear_index < m_size);
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
        CONTRACTS_CHECK(linear_index < m_size);
        return std::get<Map_Type::index()>(m_data_buffers)[linear_index];
    }

    template <concepts::MapType Map_Type, auto Buffer>
    [[nodiscard, gnu::always_inline, gnu::flatten]]
    auto get_out_patch(linear_index_t const linear_index) noexcept -> patch_t<Map_Type>&
    {
        return const_cast<patch_t<Map_Type>&>(
            std::as_const(*this).template get_out_patch<Map_Type, Buffer>(linear_index)
        );
    }

    template <concepts::MapType Map_Type, auto Buffer>
    [[nodiscard, gnu::always_inline, gnu::flatten]]
    auto get_out_patch(linear_index_t const linear_index) const noexcept
        -> patch_t<Map_Type> const&
    {
        CONTRACTS_CHECK(linear_index < m_size);
        if constexpr (Buffer == current_buffer)
            return std::get<Map_Type::index()>(m_data_buffers)[linear_index];
        if constexpr (Buffer == next_buffer)
            return std::get<Map_Type::index()>(m_next_buffers)[linear_index];
        else
            utility::error_handling::assert_unreachable();
    }

    template <concepts::MapType Map_Type, auto Buffer>
    [[nodiscard, gnu::always_inline, gnu::flatten]]
    auto get_out_patch(patch_index_t const patch_idx) noexcept -> patch_t<Map_Type>&
    {
        return const_cast<patch_t<Map_Type>&>(
            std::as_const(*this).template get_out_patch<Map_Type, Buffer>(patch_idx)
        );
    }

    template <concepts::MapType Map_Type, auto Buffer>
    [[nodiscard, gnu::always_inline, gnu::flatten]]
    auto get_out_patch(patch_index_t const patch_idx) const noexcept
        -> patch_t<Map_Type> const&
    {
        const auto linear_index = m_index_map.at(patch_idx);
        CONTRACTS_CHECK(linear_index < m_size);
        if constexpr (Buffer == current_buffer)
            return std::get<Map_Type::index()>(m_data_buffers)[linear_index];
        if constexpr (Buffer == next_buffer)
            return std::get<Map_Type::index()>(m_next_buffers)[linear_index];
        else
            utility::error_handling::assert_unreachable();
    }

#ifdef AMR_ENABLE_CUDA_AMR
    template <concepts::MapType Map_Type, auto Buffer = current_buffer>
    [[nodiscard]]
    auto get_device_buffer() noexcept -> patch_t<Map_Type>*
    {
        return const_cast<patch_t<Map_Type>*>(
            std::as_const(*this).template get_device_buffer<Map_Type, Buffer>()
        );
    }

    template <concepts::MapType Map_Type, auto Buffer = current_buffer>
    [[nodiscard]]
    auto get_device_buffer() const noexcept -> patch_t<Map_Type> const*
    {
        if constexpr (Buffer == current_buffer)
            return std::get<Map_Type::index()>(m_device_data_buffers);
        if constexpr (Buffer == next_buffer)
            return std::get<Map_Type::index()>(m_device_next_buffers);
        else
            utility::error_handling::assert_unreachable();
    }

    auto sync_current_to_device() const -> void
    {
        sync_buffer_set_to_device(m_data_buffers, m_device_data_buffers);
    }

    auto sync_current_from_device() -> void
    {
        sync_buffer_set_from_device(m_data_buffers, m_device_data_buffers);
    }

    auto sync_next_to_device() const -> void
    {
        sync_buffer_set_to_device(m_next_buffers, m_device_next_buffers);
    }

    auto sync_next_from_device() -> void
    {
        sync_buffer_set_from_device(m_next_buffers, m_device_next_buffers);
    }

    auto sync_all_to_device() const -> void
    {
        sync_current_to_device();
        sync_next_to_device();
    }

    auto sync_all_from_device() -> void
    {
        sync_current_from_device();
        sync_next_from_device();
    }
#endif

    [[nodiscard, gnu::always_inline, gnu::flatten]]
    auto get_neighbor_at(
        linear_index_t const     idx,
        patch_direction_t const& d
    ) const noexcept -> neighbor_patch_index_variant_t
    {
        CONTRACTS_CHECK_INDEX(idx, m_size);
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
                        typename ret_t::coarser{ get_linear_index_at(n.id),
                                                n.contact_quadrant }
                    };
                } },
            neighbor.data
        );
    }

    auto fragment(
        patch_index_t const node_id
#ifdef AMR_ENABLE_CUDA_AMR
        ,
        std::vector<device_transfer_task_t>* transfer_tasks = nullptr
#endif
    ) -> void
    {
        DEFAULT_SOURCE_LOG_TRACE("Fragmenting node {}", node_id.repr());
        const auto it   = find_index(node_id);
        const auto from = it.value()->second;
        CONTRACTS_CHECK(it.has_value());
        const auto start_to = m_size;
        for (size_type i = 0; i != s_nd_fanout; ++i)
        {
            const auto child_id = patch_index_t::child_of(
                node_id, static_cast<typename patch_index_t::offset_t>(i)
            );
            CONTRACTS_CHECK(!find_index(child_id).has_value());

            const auto neighbor_array =
                neighbor_utils_t::compute_child_neighbors(node_id, m_neighbors[from], i);
            append(child_id, neighbor_array);
            CONTRACTS_CHECK(m_index_map[child_id] == back_idx());
            CONTRACTS_CHECK(m_linear_index_map[back_idx()] == child_id);
        }
        enforce_symmetry_after_refinement(node_id);
#ifdef AMR_ENABLE_CUDA_AMR
        if constexpr (s_cuda_scalar_patch_compatible)
        {
            if (transfer_tasks != nullptr)
            {
                transfer_tasks->push_back(device_transfer_task_t{
                    .source_patch      = static_cast<std::int32_t>(from),
                    .destination_patch = static_cast<std::int32_t>(start_to),
                });
            }
            else
            {
                interpolate_patch(from, start_to);
            }
        }
        else
#endif
        {
            interpolate_patch(from, start_to);
        }
        m_index_map.erase(it.value());
#ifdef AMR_NDTREE_ENABLE_CHECKS
        check_index_map();
#endif
    }

    auto recombine(
        patch_index_t const parent_node_id
#ifdef AMR_ENABLE_CUDA_AMR
        ,
        std::vector<device_transfer_task_t>* transfer_tasks = nullptr
#endif
    ) -> void
    {
        using offset_t = typename patch_index_t::offset_t;
        CONTRACTS_CHECK(!find_index(parent_node_id).has_value());

        const auto child_0    = patch_index_t::child_of(parent_node_id, 0);
        const auto child_0_it = find_index(child_0);
        CONTRACTS_CHECK(child_0_it.has_value());
        const auto start = child_0_it.value()->second;

        std::array<patch_neighbors_t, s_nd_fanout> child_neighbor_arrays{};
        for (auto i = offset_t{}; i != offset_t{ s_nd_fanout }; ++i)
        {
            const auto child_i    = patch_index_t::child_of(parent_node_id, i);
            const auto child_i_it = find_index(child_i);
            CONTRACTS_CHECK(child_i_it.has_value());
            CONTRACTS_CHECK(child_i_it.value()->second == start + i);
            child_neighbor_arrays[i] = m_neighbors[m_index_map.at(child_i)];
            m_index_map.erase(child_i_it.value());
        }

        patch_neighbors_t neighbor_array =
            neighbor_utils_t::compute_parent_neighbors(child_neighbor_arrays);
        const auto to = m_size;
        append(parent_node_id, neighbor_array);
        CONTRACTS_CHECK(m_linear_index_map[back_idx()] == parent_node_id);
#ifdef AMR_ENABLE_CUDA_AMR
        if constexpr (s_cuda_scalar_patch_compatible)
        {
            if (transfer_tasks != nullptr)
            {
                transfer_tasks->push_back(device_transfer_task_t{
                    .source_patch      = static_cast<std::int32_t>(start),
                    .destination_patch = static_cast<std::int32_t>(to),
                });
            }
            else
            {
                restrict_patches(start, to);
            }
        }
        else
#endif
        {
            restrict_patches(start, to);
        }

        enforce_symmetry_after_recombining(parent_node_id, neighbor_array);
    }

    auto fragment() -> void
    {
#ifdef AMR_ENABLE_CUDA_AMR
        if constexpr (s_cuda_scalar_patch_compatible)
        {
            std::vector<device_transfer_task_t> transfer_tasks;
            transfer_tasks.reserve(m_to_refine.size());
            for (auto i = m_to_refine.size(); i > 0; --i)
            {
                fragment(m_to_refine[i - 1], &transfer_tasks);
            }
            interpolate_patches_device(transfer_tasks);
            sort_buffers();
            return;
        }
#endif
        for (auto i = m_to_refine.size(); i > 0; --i)
        {
            fragment(m_to_refine[i - 1]);
        }
        sort_buffers();
    }

    auto recombine() -> void
    {
#ifdef AMR_ENABLE_CUDA_AMR
        if constexpr (s_cuda_scalar_patch_compatible)
        {
            std::vector<device_transfer_task_t> transfer_tasks;
            transfer_tasks.reserve(m_to_coarsen.size());
            for (const auto& node_id : m_to_coarsen)
            {
                recombine(node_id, &transfer_tasks);
            }
            restrict_patches_device(transfer_tasks);
            sort_buffers();
            return;
        }
#endif
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

                    if constexpr (
                        std::is_same_v<
                            neighbor_category_t,
                            typename neighbor_patch_index_variant_t::none>
                    )
                    {
                        return;
                    }
                    else if constexpr (
                        std::is_same_v<
                            neighbor_category_t,
                            typename neighbor_patch_index_variant_t::same>
                    )
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
                    else if constexpr (
                        std::is_same_v<
                            neighbor_category_t,
                            typename neighbor_patch_index_variant_t::finer>
                    )
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
                    else if constexpr (
                        std::is_same_v<
                            neighbor_category_t,
                            typename neighbor_patch_index_variant_t::coarser>
                    )
                    {
                        CONTRACTS_CHECK(
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
                [&](auto const& neighbor_data)
                {
                    using neighbor_category_t = std::decay_t<decltype(neighbor_data)>;

                    if constexpr (
                        std::is_same_v<
                            neighbor_category_t,
                            typename neighbor_patch_index_variant_t::none>
                    )
                    {
                        return; // No neighbor to update
                    }
                    else if constexpr (
                        std::is_same_v<
                            neighbor_category_t,
                            typename neighbor_patch_index_variant_t::same>
                    )
                    {
                        // Same-level neighbor now sees the parent instead of children
                        neighbor_patch_index_variant_t new_neighbor;
                        new_neighbor.data = typename neighbor_patch_index_variant_t::same{
                            parent_node_id
                        };
                        if (const auto it = m_index_map.find(neighbor_data.id);
                            it != m_index_map.end())
                        {
                            m_neighbors[it->second][opposite_d.index()] = new_neighbor;
                        }
#ifdef AMR_NDTREE_ENABLE_CHECKS
                        else
                        {
                            // Neighbor may have been recombined earlier in the same
                            // AMR pass.
                            DEFAULT_SOURCE_LOG_WARNING(
                                "Missing same-level neighbor during recombining: " +
                                neighbor_data.id.repr()
                            );
                        }
#endif
                    }
                    else if constexpr (
                        std::is_same_v<
                            neighbor_category_t,
                            typename neighbor_patch_index_variant_t::finer>
                    )
                    {
                        // Finer neighbors now see the parent as a coarser neighbor
                        for (size_type i = 0; i != neighbor_data.num_neighbors(); i++)
                        {
                            neighbor_patch_index_variant_t new_neighbor;
                            new_neighbor
                                .data = typename neighbor_patch_index_variant_t::coarser(
                                parent_node_id,
                                neighbor_utils_t::compute_contact_quadrant(i, opposite_d)
                            );
                            m_neighbors[m_index_map.at(neighbor_data.ids[i])]
                                       [opposite_d.index()] = new_neighbor;
                        }
                    }
                    else if constexpr (
                        std::is_same_v<
                            neighbor_category_t,
                            typename neighbor_patch_index_variant_t::coarser>
                    )
                    {
                        CONTRACTS_CHECK(
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
                    [&](auto const& neighbor_data)
                    {
                        using neighbor_category_t = std::decay_t<decltype(neighbor_data)>;
                        if constexpr (
                            std::is_same_v<
                                neighbor_category_t,
                                typename neighbor_patch_index_variant_t::coarser>
                        )
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
                            if constexpr (
                                std::is_same_v<
                                    neighbor_category_t,
                                    typename neighbor_patch_index_variant_t::finer>
                            )
                            {
                                should_remove = true;
                            }
                            else if constexpr (
                                std::is_same_v<
                                    neighbor_category_t,
                                    typename neighbor_patch_index_variant_t::same>
                            )
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
#ifdef AMR_ENABLE_CUDA_AMR
        bool use_device_patch_transfers = false;
        if constexpr (s_cuda_scalar_patch_compatible)
        {
            use_device_patch_transfers = !m_to_refine.empty() || !m_to_coarsen.empty();
            if (use_device_patch_transfers)
            {
                sync_current_to_device();
            }
        }
#endif
        fragment();
        recombine();
#ifdef AMR_ENABLE_CUDA_AMR
        if constexpr (s_cuda_scalar_patch_compatible)
        {
            if (use_device_patch_transfers)
            {
                sync_current_from_device();
            }
        }
#endif
    }

    // TODO: Use this function internally where the invariant must hold rather
    // than find_index()
    [[nodiscard]]
    auto get_linear_index_at(patch_index_t const node_id) const noexcept -> linear_index_t
    {
        const auto it = m_index_map.find(node_id);
        CONTRACTS_CHECK(it != nullptr && it != m_index_map.end());
        if (it == nullptr)
        {
            DEFAULT_SOURCE_LOG_FATAL(
                "Patch index {} not found in index map", node_id.repr()
            );
            utility::error_handling::assert_unreachable();
        }
        return it->second;
    }

    [[nodiscard]]
    auto get_node_index_at(linear_index_t idx) const noexcept -> patch_index_t
    {
        CONTRACTS_CHECK(idx < m_size && "Index out of bounds in node_index_at()");
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

private:
    auto sort_buffers() noexcept -> void
    {
        compact();
        std::sort(
            m_reorder_buffer,
            &m_reorder_buffer[m_size],
            [this](auto const i, auto const j)
            { return m_linear_index_map[i] < m_linear_index_map[j]; }
        );
#ifdef AMR_ENABLE_CUDA_AMR
        std::vector<linear_index_t> device_sources(
            m_reorder_buffer, &m_reorder_buffer[m_size]
        );
#endif

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

                std::apply(
                    [dst, src](auto&... b) { ((void)(b[dst] = b[src]), ...); },
                    m_data_buffers
                );

                m_index_map[m_linear_index_map[dst]] = dst;
                m_reorder_buffer[dst]                = dst;
                dst                                  = src;
                src                                  = m_reorder_buffer[src];
                CONTRACTS_CHECK(src != dst);
            } while (src != backup_start_pos);

            m_linear_index_map[dst]     = backup_node_index;
            m_refine_status_buffer[dst] = backup_refine_status;
            m_neighbors[dst]            = backup_neighbors;

            [this, dst, &backup_patch]<std::size_t... I>(std::index_sequence<I...>)
            {
                ((void)(std::get<I>(m_data_buffers)[dst] = std::get<I>(backup_patch)),
                 ...);
            }(std::make_index_sequence<std::tuple_size_v<deconstructed_buffers_t>>{});

            m_index_map[backup_node_index] = dst;
            m_reorder_buffer[dst]          = dst;
        }
        CONTRACTS_CHECK(
            std::ranges::is_sorted(m_reorder_buffer, &m_reorder_buffer[m_size])
        );
        CONTRACTS_CHECK(is_sorted());
#ifdef AMR_ENABLE_CUDA_AMR
        permute_device_current_buffers(device_sources);
#endif
    }

    [[nodiscard]]
    auto is_sorted() const noexcept -> bool
    {
        DEFAULT_SOURCE_LOG_TRACE("Checking sort");
        if (std::ranges::is_sorted(std::span{ m_linear_index_map, m_size }, std::less{}))
        {
            for (linear_index_t i = 0; i != m_size; ++i)
            {
                CONTRACTS_CHECK(m_index_map.contains(m_linear_index_map[i]));
                if (m_index_map.at(m_linear_index_map[i]) != i)
                {
                    DEFAULT_SOURCE_LOG_ERROR("index map is not correct");
                    return false;
                }
            }
        }
        else
        {
            DEFAULT_SOURCE_LOG_ERROR("linear index is not sorted");
            return false;
        }
        return true;
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

private:
    constexpr auto patch_mapping_impl(auto&& f) noexcept -> void
    {
        amr::containers::manipulators::shaped_for<
            typename patch_layout_t::interior_iteration_control_t>(
            [this, fn = std::forward<decltype(f)>(f)](auto const& idxs)
            {
                using idxs_t = std::remove_cvref_t<decltype(idxs)>;

                idxs_t fine_patch_idxs{}; // fine patch index
                idxs_t base_fine_idxs{};  // start to the fine hypercube projection
                for (size_type d = 0; d != s_rank; ++d)
                {
                    const auto section_size =
                        patch_layout_t::data_layout_t::size(d) / s_1d_fanout;
                    fine_patch_idxs[d] = (idxs[d] - s_halo_width) / section_size;
                    base_fine_idxs[d] =
                        ((idxs[d] - s_halo_width) % section_size) * s_1d_fanout +
                        s_halo_width;
                }
                const auto fine_patch_idx =
                    projection_hypercube_t::layout_t::linear_index(fine_patch_idxs);
                const auto base_fine_idx =
                    patch_layout_t::padded_layout_t::linear_index(base_fine_idxs);
                // flattened hypercube of the idxs in the projected fine patch
                const auto fine_linear_idxs =
                    utils::patches::detail::hypercube_offset<patch_layout_t, 2>(
                        base_fine_idx
                    );
                const auto coarse_idx =
                    patch_layout_t::padded_layout_t::linear_index(idxs);

                fn(fine_patch_idx, fine_linear_idxs, coarse_idx);
            }
        );
    }

    constexpr auto restrict_patches(
        linear_index_t const start_from,
        linear_index_t const to
    ) noexcept -> void
    {
        DEFAULT_SOURCE_LOG_TRACE("Patch restriction");

        patch_mapping_impl(
            [this,
             start_from,
             to](auto fine_patch_idx, auto const& fine_linear_idxs, auto to_idx)
            {
                std::apply(
                    [to,
                     to_idx = to_idx,
                     from   = start_from + fine_patch_idx,
                     fine_linear_idxs](auto&... b)
                    {
                        ((intergrid_operator_t::restriction(
                             b[to], to_idx, b[from], fine_linear_idxs
                         )),
                         ...);
                    },
                    m_data_buffers
                );
            }
        );
    }

    constexpr auto interpolate_patch(
        linear_index_t const from,
        linear_index_t const start_to
    ) noexcept -> void
    {
        DEFAULT_SOURCE_LOG_TRACE("Patch interpolation");

        patch_mapping_impl(
            [this,
             from,
             start_to](auto fine_patch_idx, auto const& fine_linear_idxs, auto from_idx)
            {
                std::apply(
                    [to      = start_to + fine_patch_idx,
                     to_idxs = fine_linear_idxs,
                     from_idx,
                     from](auto&... b)
                    {
                        ((intergrid_operator_t::interpolation(
                             b[to], to_idxs, b[from], from_idx
                         )),
                         ...);
                    },
                    m_data_buffers
                );
            }
        );
    }

public:
    constexpr auto swap_buffers() noexcept -> void
    {
        DEFAULT_SOURCE_LOG_TRACE("Swap buffers");
        std::swap(m_data_buffers, m_next_buffers);
#ifdef AMR_ENABLE_CUDA_AMR
        std::swap(m_device_data_buffers, m_device_next_buffers);
#endif
#ifndef NDEBUG
        std::apply(
            [this](auto&&... b)
            {
                (std::fill_n(
                     (unwrap_value_t<decltype(b)>*)b,
                     m_size * patch_layout_t::padded_layout_t::flat_size(),
                     -1000
                 ),
                 ...);
            },
            m_next_buffers
        );
#endif
    }

    auto halo_exchange_update_cpu() noexcept -> void
    {
        DEFAULT_SOURCE_LOG_TRACE("Performing halo exchange");
        auto const r = std::views::iota(size_type{}, m_size);
        std::for_each(
            AMR_EXECUTION_POLICY,
            std::cbegin(r),
            std::cend(r),
            [this](auto const i)
            {
                utils::patches::halo_apply<
                    halo_exchange_operator_impl_t,
                    patch_direction_t>(*this, i);
            }
        );
    }

#ifdef AMR_ENABLE_CUDA_AMR
private:
    auto build_halo_exchange_metadata() const -> std::vector<amr::cuda::halo_direction_metadata>
    {
        using metadata_t = amr::cuda::halo_direction_metadata;
        std::vector<metadata_t> metadata(m_size * patch_direction_t::elements());

        for (linear_index_t idx = 0; idx != m_size; ++idx)
        {
            for (auto d = patch_direction_t::first(); d != patch_direction_t::sentinel();
                 d.advance())
            {
                auto& out = metadata[idx * patch_direction_t::elements() + d.index()];
                const auto neighbor = neighbor_linear_index(get_neighbor_at(idx, d));
                std::visit(
                    [&out](auto const& n)
                    {
                        using neighbor_t = std::decay_t<decltype(n)>;
                        if constexpr (std::is_same_v<
                                          neighbor_t,
                                          typename neighbor_linear_index_variant_t::none>)
                        {
                            out.relation = amr::cuda::halo_neighbor_relation::none;
                        }
                        else if constexpr (std::is_same_v<
                                               neighbor_t,
                                               typename neighbor_linear_index_variant_t::same>)
                        {
                            out.relation = amr::cuda::halo_neighbor_relation::same;
                            out.neighbor = static_cast<std::int32_t>(n.id);
                        }
                        else if constexpr (std::is_same_v<
                                               neighbor_t,
                                               typename neighbor_linear_index_variant_t::finer>)
                        {
                            out.relation = amr::cuda::halo_neighbor_relation::finer;
                            for (std::size_t i = 0; i != n.num_neighbors(); ++i)
                            {
                                out.finer_ids[i] = static_cast<std::int32_t>(n.ids[i]);
                            }
                        }
                        else if constexpr (std::is_same_v<
                                               neighbor_t,
                                               typename neighbor_linear_index_variant_t::coarser>)
                        {
                            out.relation = amr::cuda::halo_neighbor_relation::coarser;
                            out.neighbor = static_cast<std::int32_t>(n.id);
                            for (std::size_t i = 0; i != s_rank; ++i)
                            {
                                out.quadrant[i] =
                                    static_cast<std::int32_t>(n.contact_quadrant[i]);
                            }
                        }
                    },
                    neighbor.data
                );
            }
        }

        return metadata;
    }

    template <concepts::MapType Map_Type>
    auto halo_exchange_update_device_map(
        std::vector<amr::cuda::halo_direction_metadata> const& metadata
    ) -> void
    {
        using patch_value_t = patch_t<Map_Type>;
        static_assert(
            std::is_same_v<typename Map_Type::type, double> &&
            sizeof(patch_value_t) == patch_layout_t::flat_size() * sizeof(double)
        );

        auto* device_patch_data =
            reinterpret_cast<double*>(get_device_buffer<Map_Type>());

        std::array<std::size_t, 3> padded_sizes{ 1, 1, 1 };
        std::array<std::size_t, 3> padded_strides{ 1, 1, 1 };
        std::array<std::size_t, 3> data_sizes{ 1, 1, 1 };
        for (std::size_t d = 0; d != s_rank; ++d)
        {
            padded_sizes[d] = patch_layout_t::padded_layout_t::sizes()[d];
            padded_strides[d] = patch_layout_t::padded_layout_t::strides()[d];
            data_sizes[d] = patch_layout_t::data_layout_t::sizes()[d];
        }

        amr::cuda::halo_exchange_scalar_patches_inplace(
            device_patch_data,
            metadata.data(),
            metadata.size(),
            amr::cuda::halo_exchange_launch_config{
                .num_patches     = m_size,
                .patch_flat_size = patch_layout_t::flat_size(),
                .rank            = s_rank,
                .halo_width      = s_halo_width,
                .fanout_1d       = s_1d_fanout,
                .padded_sizes    = padded_sizes,
                .padded_strides  = padded_strides,
                .data_sizes      = data_sizes,
            }
        );
    }

    auto halo_exchange_update_device() -> void
    {
        sync_current_to_device();
        const auto metadata = build_halo_exchange_metadata();
        [&metadata, this]<std::size_t... I>(std::index_sequence<I...>)
        {
            (halo_exchange_update_device_map<
                 std::tuple_element_t<I, deconstructed_raw_map_types_t>>(metadata),
             ...);
        }(std::make_index_sequence<std::tuple_size_v<deconstructed_raw_map_types_t>>{});
        sync_current_from_device();
    }

    [[nodiscard]]
    auto make_intergrid_transfer_launch_config() const
        -> amr::cuda::intergrid_transfer_launch_config
    {
        std::array<std::size_t, 3> padded_strides{ 1, 1, 1 };
        std::array<std::size_t, 3> data_sizes{ 1, 1, 1 };
        std::array<std::size_t, 3> data_strides{ 1, 1, 1 };
        for (std::size_t d = 0; d != s_rank; ++d)
        {
            padded_strides[d] = patch_layout_t::padded_layout_t::strides()[d];
            data_sizes[d]     = patch_layout_t::data_layout_t::sizes()[d];
            data_strides[d]   = patch_layout_t::data_layout_t::strides()[d];
        }

        return amr::cuda::intergrid_transfer_launch_config{
            .patch_flat_size = patch_layout_t::flat_size(),
            .data_flat_size  = patch_layout_t::data_layout_t::flat_size(),
            .rank            = s_rank,
            .halo_width      = s_halo_width,
            .fanout_1d       = s_1d_fanout,
            .padded_strides  = padded_strides,
            .data_sizes      = data_sizes,
            .data_strides    = data_strides,
        };
    }

    template <concepts::MapType Map_Type>
    auto interpolate_patches_device_map(
        std::vector<device_transfer_task_t> const&         transfer_tasks,
        amr::cuda::intergrid_transfer_launch_config const& config
    ) -> void
    {
        using patch_value_t = patch_t<Map_Type>;
        static_assert(
            std::is_same_v<typename Map_Type::type, double> &&
            sizeof(patch_value_t) == patch_layout_t::flat_size() * sizeof(double)
        );

        auto* device_patch_data =
            reinterpret_cast<double*>(get_device_buffer<Map_Type>());
        amr::cuda::interpolate_scalar_patches_inplace(
            device_patch_data,
            transfer_tasks.data(),
            transfer_tasks.size(),
            config
        );
    }

    template <concepts::MapType Map_Type>
    auto restrict_patches_device_map(
        std::vector<device_transfer_task_t> const&         transfer_tasks,
        amr::cuda::intergrid_transfer_launch_config const& config
    ) -> void
    {
        using patch_value_t = patch_t<Map_Type>;
        static_assert(
            std::is_same_v<typename Map_Type::type, double> &&
            sizeof(patch_value_t) == patch_layout_t::flat_size() * sizeof(double)
        );

        auto* device_patch_data =
            reinterpret_cast<double*>(get_device_buffer<Map_Type>());
        amr::cuda::restrict_scalar_patches_inplace(
            device_patch_data,
            transfer_tasks.data(),
            transfer_tasks.size(),
            config
        );
    }

    auto interpolate_patches_device(
        std::vector<device_transfer_task_t> const& transfer_tasks
    ) -> void
    {
        if (transfer_tasks.empty())
        {
            return;
        }

        const auto config = make_intergrid_transfer_launch_config();
        [&transfer_tasks, &config, this]<std::size_t... I>(std::index_sequence<I...>)
        {
            (interpolate_patches_device_map<
                 std::tuple_element_t<I, deconstructed_raw_map_types_t>>(
                 transfer_tasks, config
             ),
             ...);
        }(std::make_index_sequence<std::tuple_size_v<deconstructed_raw_map_types_t>>{});
    }

    auto restrict_patches_device(
        std::vector<device_transfer_task_t> const& transfer_tasks
    ) -> void
    {
        if (transfer_tasks.empty())
        {
            return;
        }

        const auto config = make_intergrid_transfer_launch_config();
        [&transfer_tasks, &config, this]<std::size_t... I>(std::index_sequence<I...>)
        {
            (restrict_patches_device_map<
                 std::tuple_element_t<I, deconstructed_raw_map_types_t>>(
                 transfer_tasks, config
             ),
             ...);
        }(std::make_index_sequence<std::tuple_size_v<deconstructed_raw_map_types_t>>{});
    }

public:
#endif
    auto halo_exchange_update() noexcept -> void

    {
#ifdef AMR_ENABLE_CUDA_AMR
        if constexpr (s_cuda_scalar_halo_compatible)
        {
            halo_exchange_update_device();
        }
        else
        {
            halo_exchange_update_cpu();
        }
#else
        halo_exchange_update_cpu();
#endif
    }

    [[nodiscard]]
    auto get_refine_status(const linear_index_t i) const noexcept -> refine_status_t
    {
        CONTRACTS_CHECK(i < m_size);
        return m_refine_status_buffer[i];
    }

private:
    [[nodiscard]]
    auto is_refine_elegible(const linear_index_t i) const noexcept -> bool
    {
        const auto node_id = m_linear_index_map[i];
        CONTRACTS_CHECK(m_index_map.contains(node_id));
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
            // TODO: Maybe do this an CONTRACTS_CHECK rather
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
#ifdef AMR_ENABLE_CUDA_AMR
        std::vector<linear_index_t> device_sources;
        device_sources.reserve(m_size);
#endif
        for (linear_index_t head = 0; head < m_size; ++head)
        {
            const auto node_id = m_linear_index_map[head];
            if (m_index_map.contains(node_id))
            {
#ifdef AMR_ENABLE_CUDA_AMR
                device_sources.push_back(head);
#endif
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
#ifdef AMR_ENABLE_CUDA_AMR
        permute_device_current_buffers(device_sources);
#endif
    }

    [[gnu::always_inline, gnu::flatten]]
    auto block_buffer_swap(linear_index_t const i, linear_index_t const j) noexcept
        -> void
    {
        CONTRACTS_CHECK(i < m_size);
        CONTRACTS_CHECK(j < m_size);
        if (i == j)
        {
            return;
        }

        CONTRACTS_CHECK(m_linear_index_map[i] != m_linear_index_map[j]);
        std::swap(m_linear_index_map[i], m_linear_index_map[j]);
        std::swap(m_refine_status_buffer[i], m_refine_status_buffer[j]);
        std::swap(m_neighbors[i], m_neighbors[j]);

        std::apply(
            [i, j](auto&... b) { ((void)std::swap(b[i], b[j]), ...); }, m_data_buffers
        );
    }

#ifdef AMR_ENABLE_CUDA_AMR
    template <std::size_t... I>
    auto sync_buffer_set_to_device_impl(
        deconstructed_buffers_t const& host_buffers,
        deconstructed_buffers_t const& device_buffers,
        std::index_sequence<I...>
    ) const -> void
    {
        (
            (void)amr::cuda::copy_host_to_device(
                static_cast<void*>(std::get<I>(device_buffers)),
                static_cast<void const*>(std::get<I>(host_buffers)),
                m_size * sizeof(std::remove_pointer_t<std::remove_reference_t<decltype(std::get<I>(host_buffers))>>)
            ),
            ...
        );
    }

    template <std::size_t... I>
    auto sync_buffer_set_from_device_impl(
        deconstructed_buffers_t&       host_buffers,
        deconstructed_buffers_t const& device_buffers,
        std::index_sequence<I...>
    ) const -> void
    {
        (
            (void)amr::cuda::copy_device_to_host(
                static_cast<void*>(std::get<I>(host_buffers)),
                static_cast<void const*>(std::get<I>(device_buffers)),
                m_size * sizeof(std::remove_pointer_t<std::remove_reference_t<decltype(std::get<I>(host_buffers))>>)
            ),
            ...
        );
    }

    auto sync_buffer_set_to_device(
        deconstructed_buffers_t const& host_buffers,
        deconstructed_buffers_t const& device_buffers
    ) const -> void
    {
        sync_buffer_set_to_device_impl(
            host_buffers,
            device_buffers,
            std::make_index_sequence<std::tuple_size_v<deconstructed_buffers_t>>{}
        );
    }

    auto sync_buffer_set_from_device(
        deconstructed_buffers_t&       host_buffers,
        deconstructed_buffers_t const& device_buffers
    ) const -> void
    {
        sync_buffer_set_from_device_impl(
            host_buffers,
            device_buffers,
            std::make_index_sequence<std::tuple_size_v<deconstructed_buffers_t>>{}
        );
    }

    template <std::size_t... I>
    auto permute_device_buffer_set_impl(
        deconstructed_buffers_t&        device_buffers,
        std::vector<linear_index_t> const& sources,
        std::index_sequence<I...>
    ) const -> void
    {
        (
            (void)amr::cuda::permute_patches_inplace(
                static_cast<void*>(std::get<I>(device_buffers)),
                sources.size(),
                sizeof(std::remove_pointer_t<std::remove_reference_t<decltype(std::get<I>(device_buffers))>>),
                sources.data(),
                sources.size()
            ),
            ...
        );
    }

    auto permute_device_current_buffers(
        std::vector<linear_index_t> const& sources
    ) -> void
    {
        if (sources.empty())
        {
            return;
        }

        permute_device_buffer_set_impl(
            m_device_data_buffers,
            sources,
            std::make_index_sequence<std::tuple_size_v<deconstructed_buffers_t>>{}
        );
    }
#endif

#ifdef AMR_NDTREE_ENABLE_CHECKS
    auto check_index_map() const noexcept -> void
    {
        if (m_index_map.size() > m_size)
        {
            DEFAULT_SOURCE_LOG_ERROR("morton index map size exceeds tree size");
            CONTRACTS_CHECK(false);
            return;
        }

        for (const auto& [node_idx, linear_idx] : m_index_map)
        {
            if (m_linear_index_map[linear_idx] != node_idx)
            {
                DEFAULT_SOURCE_LOG_ERROR("morton index map is incorrect");
                CONTRACTS_CHECK(false);
                return;
            }
        }
    }
#endif

private:
    index_map_t                m_index_map;
    deconstructed_buffers_t    m_data_buffers;
    deconstructed_buffers_t    m_next_buffers;
#ifdef AMR_ENABLE_CUDA_AMR
    deconstructed_buffers_t    m_device_data_buffers{};
    deconstructed_buffers_t    m_device_next_buffers{};
#endif
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
