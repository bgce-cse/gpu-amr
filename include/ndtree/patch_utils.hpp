#ifndef AMR_INCLUDED_PATCH_UTILS
#define AMR_INCLUDED_PATCH_UTILS

#include "containers/container_manipulations.hpp"
#include "containers/container_utils.hpp"
#include "containers/static_tensor.hpp"
#include "intergrid_operator.hpp"
#include "ndconcepts.hpp"
#include "ndutils.hpp"
#include "utility/compile_time_utility.hpp"
#include "utility/contracts.hpp"
#include "utility/logging.hpp"
#include <algorithm>

namespace amr::ndt::utils
{

namespace patches
{

[[nodiscard]]
consteval auto multiples_of(
    std::ranges::input_range auto const& r,
    std::integral auto const             x
) noexcept -> bool
    requires std::convertible_to<
        std::ranges::range_value_t<std::remove_cvref_t<decltype(r)>>,
        std::remove_const_t<decltype(x)>>
{
    return std::ranges::all_of(r, [x](auto const& e) { return (e) % x == 0; });
}

template <concepts::PatchLayout Layout>
[[nodiscard]]
constexpr auto is_halo_cell(typename Layout::index_t linear_index) noexcept -> bool
{
    using patch_layout_t             = Layout;
    using layout_t                   = typename patch_layout_t::padded_layout_t;
    using size_type                  = layout_t::size_type;
    constexpr auto        rank       = layout_t::rank();
    constexpr auto const& strides    = layout_t::strides();
    constexpr auto const& sizes      = layout_t::sizes();
    constexpr size_type   halo_width = patch_layout_t::halo_width();

    CONTRACTS_CHECK_INDEX(linear_index, layout_t::flat_size());

    for (auto j = decltype(rank){}; j != rank; ++j)
    {
        const auto relative_idx = static_cast<size_type>(linear_index / strides[j]);
        CONTRACTS_CHECK(relative_idx < sizes[j]);
        if (relative_idx < halo_width || relative_idx >= sizes[j] - halo_width)
        {
            return true;
        }
        linear_index %= strides[j];
    }
    return false;
}

template <concepts::PatchLayout Patch_Layout, std::integral auto Fanout>
[[nodiscard, deprecated]]
constexpr auto fragmentation_patch_maps() noexcept
    -> containers::utils::types::tensor::hypercube_t<
        containers::static_tensor<
            typename Patch_Layout::index_t,
            typename Patch_Layout::padded_layout_t>,
        Fanout,
        Patch_Layout::rank()>
{
    using patch_layout_t  = Patch_Layout;
    constexpr auto fanout = static_cast<typename patch_layout_t::index_t>(Fanout);

    using layout_t = typename patch_layout_t::padded_layout_t;
    using index_t  = typename patch_layout_t::index_t;
    using tensor_t = typename containers::static_tensor<index_t, layout_t>;
    using patch_shape_t =
        containers::utils::types::tensor::hypercube_t<tensor_t, fanout, tensor_t::rank()>;
    patch_shape_t to{};

    constexpr index_t halo_width = patch_layout_t::halo_width();
    constexpr auto&   strides    = layout_t::strides();
    constexpr auto&   data_shape = patch_layout_t::data_layout_t::sizes();

    auto idx = typename tensor_t::multi_index_t{};
    do
    {
        const auto linear_idx = layout_t::linear_index(idx);
        const auto is_halo    = is_halo_cell<patch_layout_t>(linear_idx);
        const auto        offset     = is_halo
            ? [&idx]()
            {
                index_t ret{};
                for(auto i = index_t{}; i != layout_t::rank(); ++i)
                {
                    const auto inc = (idx[i] - halo_width + data_shape[i])
                                   % data_shape[i] + halo_width;
                    ret += inc * strides[i];
                }
                return ret;
            }()
            : std::transform_reduce(
                std::cbegin(idx),
                std::cend(idx),
                std::cbegin(strides),
                index_t{},
                std::plus{},
                [](auto const i, auto const s)
                {
                    CONTRACTS_CHECK(i >= halo_width);
                    return ((i - halo_width) / fanout) * s;
                }
            );
        auto out_patch_idx = typename patch_shape_t::multi_index_t{};
        do
        {
            const auto linear_out_idx      = patch_shape_t::linear_index(out_patch_idx);
            const auto base                = is_halo ? index_t{} : [&out_patch_idx]()
            {
                index_t ret{};
                for (index_t i = 0; i != layout_t::rank(); ++i)
                {
                    CONTRACTS_CHECK(data_shape[i] % fanout == 0);
                    ret += (out_patch_idx[i] * (data_shape[i] / fanout) + halo_width) *
                           strides[i];
                }
                return ret;
            }();
            to[linear_out_idx][linear_idx] = offset + base;
        } while (out_patch_idx.increment());
    } while (idx.increment());
    return to;
}

namespace detail
{

template <
    concepts::HaloExchangeOperator Halo_Exchange_Operator,
    concepts::Direction auto       D,
    concepts::MapType              T>
constexpr auto halo_apply_section_impl(
    auto&&                                                             tree,
    typename std::remove_cvref_t<decltype(tree)>::linear_index_t const idx,
    typename std::remove_cvref_t<decltype(tree)>::neighbor_linear_index_variant_t const&
        n_idx,
    auto&&... args
) noexcept -> void
    requires concepts::TreeType<std::remove_cvref_t<decltype(tree)>>
{
    using tree_t                = std::remove_cvref_t<decltype(tree)>;
    using n_linear_idx_varant_t = typename tree_t::neighbor_linear_index_variant_t;
    using patch_layout_t        = typename tree_t::patch_layout_t;
    auto& p_i = std::forward<decltype(tree)>(tree).template get_patch<T>(idx);

    DEFAULT_SOURCE_LOG_TRACE("{} halo exchange in direction {}", n_idx.repr(), D.repr());
    std::visit(
        utils::overloads{
            // None impl
            [&p_i, &args...](typename n_linear_idx_varant_t::none const&)
            {
                containers::manipulators::for_each<
                    typename patch_layout_t::template halo_iteration_control_t<D>>(
                    p_i.data(),
                    Halo_Exchange_Operator::boundary,
                    D,
                    std::forward<decltype(args)>(args)...
                );
            },
            // Same impl
            [&tree, &p_i, &args...](typename n_linear_idx_varant_t::same const& neighbor)
            {
                const auto& p_n =
                    std::forward<decltype(tree)>(tree).template get_patch<T>(neighbor.id);

                containers::manipulators::for_each<
                    typename patch_layout_t::template halo_iteration_control_t<D>>(
                    p_i.data(),
                    Halo_Exchange_Operator::same,
                    p_n.data(),
                    D,
                    std::forward<decltype(args)>(args)...
                );
            },
            // Finer impl
            [&tree, &p_i, &n_idx, &idx, &args...](
                typename n_linear_idx_varant_t::finer const& neighbor
            )
            {
                using neighbor_t = std::remove_cvref_t<decltype(neighbor)>;
                using patch_t =
                    typename std::remove_cvref_t<decltype(tree)>::template patch_t<T>;

                Halo_Exchange_Operator::s_debug_current_patch_linear_idx = idx;

                const auto p_neighbors = utility::compile_time_utility::array_factory<
                    std::reference_wrapper<typename patch_t::container_t>,
                    neighbor_t::num_neighbors()>(
                    [&tree](auto const i, auto const& ids)
                    {
                        return std::ref(
                            std::forward<decltype(tree)>(tree)
                                .template get_patch<T>(ids[i])
                                .data()
                        );
                    },
                    neighbor.ids
                );

                containers::manipulators::for_each<
                    typename patch_layout_t::template halo_iteration_control_t<D>>(
                    p_i.data(),
                    Halo_Exchange_Operator::finer,
                    p_neighbors,
                    neighbor.ids,
                    D,
                    std::forward<decltype(args)>(args)...
                );
            },
            // Coarser impl
            [&tree,
             &p_i,
             &args...](typename n_linear_idx_varant_t::coarser const& neighbor)
            {
                const auto& p_n =
                    std::forward<decltype(tree)>(tree).template get_patch<T>(neighbor.id);

                containers::manipulators::for_each<
                    typename patch_layout_t::template halo_iteration_control_t<D>>(
                    p_i.data(),
                    Halo_Exchange_Operator::coarser,
                    p_n.data(),
                    D,
                    neighbor.contact_quadrant,
                    std::forward<decltype(args)>(args)...
                );
            } },
        n_idx.data
    );
}

template <
    concepts::HaloExchangeOperator Halo_Exchange_Operator,
    concepts::Direction auto       D>
constexpr auto halo_apply_unroll_impl(
    auto&&                                                             tree,
    typename std::remove_cvref_t<decltype(tree)>::linear_index_t const idx,
    auto&&... args
) noexcept -> void
    requires concepts::TreeType<std::remove_cvref_t<decltype(tree)>>
{
    if constexpr (D == decltype(D)::sentinel())
    {
    }
    else
    {
        using tree_t      = std::remove_cvref_t<decltype(tree)>;
        using map_types_t = typename tree_t::deconstructed_raw_map_types_t;
        auto const& n_idx = tree.neighbor_linear_index(tree.get_neighbor_at(idx, D));
        [&n_idx]<std::size_t... I>(
            std::index_sequence<I...>, auto&& t, auto i, auto&&... a
        )
        {
            (halo_apply_section_impl<
                 Halo_Exchange_Operator,
                 D,
                 std::tuple_element_t<I, map_types_t>>(
                 std::forward<decltype(t)>(t), i, n_idx, std::forward<decltype(a)>(a)...
             ),
             ...);
        }(std::make_index_sequence<std::tuple_size_v<map_types_t>>{},
          std::forward<decltype(tree)>(tree),
          idx,
          std::forward<decltype(args)>(args)...);

        // Recursive call
        detail::halo_apply_unroll_impl<Halo_Exchange_Operator, decltype(D)::advance(D)>(
            std::forward<decltype(tree)>(tree), idx, std::forward<decltype(args)>(args)...
        );
    }
}

template <concepts::PatchLayout Patch_Layout, std::integral auto N>
static constexpr auto hypercube_offset(typename Patch_Layout::index_t idx) noexcept
    -> std::array<
        typename Patch_Layout::index_t,
        utility::cx_functions::pow(N, Patch_Layout::rank())>
{
    using patch_t  = Patch_Layout;
    using index_t  = typename patch_t::index_t;
    using layout_t = typename patch_t::padded_layout_t;
    using loop_control_t =
        containers::control::loop_control<typename layout_t::shape_t, 0, N, 1>;
    static constexpr auto  dim     = Patch_Layout::rank();
    static constexpr auto  k       = utility::cx_functions::pow(N, dim);
    static constexpr auto& strides = layout_t::strides();
    using ret_t                    = std::array<index_t, k>;
    ret_t   ret{};
    index_t out_i = 0;
    amr::containers::manipulators::shaped_for<loop_control_t>(
        [](auto& out, index_t out_idx, auto& oi, auto const& idxs)
        {
            // std::cout << "lidx: " << oi << ", out_idx: " << out_idx << '\n';
            for (index_t i = 0; i != dim; ++i)
            {
                // std::cout << i << " -> idxs[i]: " << idxs[i]
                //           << ", stride[-i]: " << strides[dim - 1 - i] << '\n';
                out_idx += idxs[i] * strides[i];
            }
            out[oi++] = out_idx;
        },
        ret,
        idx,
        out_i
    );
    return ret;
}

} // namespace detail

template <
    concepts::HaloExchangeOperator Halo_Exchange_Operator,
    concepts::Direction            D_Type>
constexpr auto halo_apply(
    auto&&                                                             tree,
    typename std::remove_cvref_t<decltype(tree)>::linear_index_t const idx,
    auto&&... args
) noexcept -> void
    requires concepts::TreeType<std::remove_cvref_t<decltype(tree)>>
{
    detail::halo_apply_unroll_impl<Halo_Exchange_Operator, D_Type::first()>(
        std::forward<decltype(tree)>(tree), idx, std::forward<decltype(args)>(args)...
    );
}

template <
    concepts::PatchIndex        Patch_Index,
    concepts::PatchLayout       Patch_Layout,
    concepts::IntergridOperator Intergrid_Operator>
struct halo_exchange_impl_t
{
    using patch_layout_t       = Patch_Layout;
    using patch_index_t        = Patch_Index;
    using index_t              = typename patch_layout_t::index_t;
    using intergrid_operator_t = Intergrid_Operator;

    // TODO: Revisit this after propperly typing patch_index_t
    static constexpr auto s_halo_width =
        static_cast<index_t>(patch_layout_t::halo_width());
    static constexpr auto s_rank      = static_cast<index_t>(patch_layout_t::rank());
    static constexpr auto s_1d_fanout = static_cast<index_t>(patch_index_t::fanout());
    static constexpr auto s_nd_fanout = static_cast<index_t>(patch_index_t::nd_fanout());
    static constexpr auto s_sizes     = patch_layout_t::data_layout_t::sizes();
    inline static thread_local index_t s_debug_current_patch_linear_idx{};

    using projection_hypercube_t = amr::containers::utils::types::tensor::hypercube_t<
        typename patch_layout_t::padded_layout_t::index_t,
        s_1d_fanout,
        s_rank>;

    /// Compute child_offset using the same convention as ndtree's
    /// interpolate_patch / restrict_patches.  Takes a padded multi-index.
    static constexpr std::size_t
        ndtree_child_offset(auto const& padded_multi_idx) noexcept
    {
        static constexpr auto& padded_strides =
            patch_layout_t::padded_layout_t::strides();

        // Build padded linear index from multi-index
        index_t padded_linear = 0;
        for (index_t d = 0; d < s_rank; ++d)
            padded_linear += padded_multi_idx[d] * padded_strides[d];

        // Same loop as ndtree: padded strides + data sizes
        std::size_t offset = 0;
        for (index_t d = 0; d < s_rank; ++d)
        {
            auto coord = static_cast<std::size_t>(
                (padded_linear / padded_strides[d]) % s_sizes[d]
            );
            auto fine_in_coarse = coord % s_1d_fanout;
            offset              = offset * s_1d_fanout + fine_in_coarse;
        }
        return (s_nd_fanout - 1) - offset;
    }

    struct boundary_t
    {
        static constexpr auto operator()(
            [[maybe_unused]] auto const& p_i,
            [[maybe_unused]] auto const& direction,
            [[maybe_unused]] auto&&... args
        ) noexcept -> void
        {
            DEFAULT_SOURCE_LOG_WARNING("Boundary halo exchange not implemented");
        }
    };

    struct same_t
    {
        static constexpr auto operator()(
            auto&       self_patch,
            auto const& other_patch,
            auto const& direction,
            auto const& idxs,
            [[maybe_unused]] auto&&... args
        ) noexcept -> void
        {
            const auto dim       = direction.dimension();
            const auto positive  = direction.is_positive();
            auto       from_idxs = idxs;
            from_idxs[dim]       = positive ? (from_idxs[dim] - index_t{ s_sizes[dim] })
                                            : (from_idxs[dim] + index_t{ s_sizes[dim] });
            self_patch[idxs]     = other_patch[from_idxs];
        }
    };

    struct finer_t
    {
        static constexpr auto operator()(
            auto&                                     current_patch,
            std::ranges::contiguous_range auto const& neighbor_patches,
            auto const&                               neighbor_patch_ids,
            auto const&                               direction,
            auto const&                               idxs,
            [[maybe_unused]] auto&&... args
        ) noexcept -> void
        {
            const auto dim      = direction.dimension();
            const auto positive = direction.is_positive();

            index_t patch_linear_idx = 0;
            index_t stride           = 1;
            auto    base_fine_idxs   = idxs;

            base_fine_idxs[dim] = positive ? (idxs[dim] + index_t{ s_sizes[dim] })
                                           : (idxs[dim] - index_t{ s_sizes[dim] });

            for (index_t d = 0; d < s_rank; ++d)
            {
                const auto section_size = s_sizes[d] / s_1d_fanout;
                const auto from_d       = base_fine_idxs[d];
                const auto local_d      = from_d - s_halo_width;

                if (d != dim)
                {
                    patch_linear_idx += (local_d / section_size) * stride;
                    stride *= s_1d_fanout;
                }
                base_fine_idxs[d] = (local_d % section_size) * s_1d_fanout + s_halo_width;
            }

            auto& fine_patch = neighbor_patches[patch_linear_idx].get();
            [[maybe_unused]]
            const auto fine_patch_lidx = neighbor_patch_ids[patch_linear_idx];

            const auto base_linear_idx =
                patch_layout_t::padded_layout_t::linear_index(base_fine_idxs);
            const auto fine_linear_idxs =
                detail::hypercube_offset<patch_layout_t, 2>(base_linear_idx);
            const auto to_idx = patch_layout_t::padded_layout_t::linear_index(idxs);
            // DEFAULT_SOURCE_LOG_INFO(
            //     "coarse_patch_lidx={}, halo_cell_lidx={}, direction={}{} , "
            //     "fine_patch_lidx={}, fine_patch_idx={}, base_linear_idx={}",
            //     s_debug_current_patch_linear_idx,
            //     to_idx,
            //     signed_dir,
            //     dim,
            //     fine_patch_lidx,
            //     fine_patch_idx,
            //     base_linear_idx
            // );
            // for (index_t fi = index_t{}; fi != s_nd_fanout; ++fi)
            // {
            //     DEFAULT_SOURCE_LOG_INFO(
            //         "finer_cells_lidx[{}]={}", fi, fine_linear_idxs[fi]
            //     );
            // }
            intergrid_operator_t::restriction(
                current_patch, to_idx, fine_patch, fine_linear_idxs
            );
        }
    };

    struct coarser_t
    {
        static constexpr auto operator()(
            auto&       self_patch,
            auto const& other_patch,
            auto const& direction,
            auto const& contact_quadrant,
            auto const& idxs,
            [[maybe_unused]] auto&&... args
        ) noexcept -> void
        {
            const auto dim      = direction.dimension();
            const auto positive = direction.is_positive();

            CONTRACTS_CHECK(
                (positive && (contact_quadrant[dim] == index_t{})) ||
                (!positive && (contact_quadrant[dim] == s_1d_fanout - index_t{ 1 }))
            );

            std::array<index_t, s_rank> from_idxs{};
            for (auto i = index_t{}; i != s_rank; ++i)
            {
                CONTRACTS_CHECK_INDEX(contact_quadrant[i], s_1d_fanout);
                CONTRACTS_CHECK(
                    i == dim ? ((!positive && (idxs[i] < s_halo_width)) ||
                                (positive && (idxs[i] >= s_halo_width + s_sizes[i])))
                             : ((idxs[i] >= s_halo_width) &&
                                (idxs[i] < s_halo_width + s_sizes[i]))
                );

                const auto cells_per_block = (s_sizes[i] / s_1d_fanout);
                const auto fine_mapped_idx =
                    (i == dim) ? (positive ? idxs[i] - s_sizes[i] - s_halo_width
                                           : idxs[i] + s_sizes[i] - s_halo_width)
                               : idxs[i] - s_halo_width;
                from_idxs[i] = s_halo_width + (contact_quadrant[i] * cells_per_block) +
                               fine_mapped_idx / s_1d_fanout;
                // const auto cells_per_block = (s_sizes[i] / s_1d_fanout);
                // const auto block_offset =
                //     (i == dim && !positive)
                //         ? (cells_per_block - s_halo_width / s_1d_fanout)
                //         : index_t{};
                // const auto idx_offset =
                //     i == dim ? (direction.is_positive() ? s_sizes[i] + s_halo_width
                //                                         : index_t{})
                //              : s_halo_width;
                // CONTRACTS_CHECK_INDEX(
                //     idxs[i] - idx_offset, i == dim ? s_halo_width : s_sizes[i]
                // );
                // CONTRACTS_CHECK_INDEX(contact_quadrant[i], s_1d_fanout);
                // from_idxs[i] = s_halo_width + (contact_quadrant[i] * cells_per_block) +
                //                block_offset + (idxs[i] - idx_offset) / s_1d_fanout;
                CONTRACTS_CHECK_INDEX(from_idxs[i] - s_halo_width, s_sizes[i]);
            }

            const auto child_offset =
                projection_hypercube_t::linear_index(contact_quadrant);
            const auto to_idx = patch_layout_t::padded_layout_t::linear_index(idxs);
            const auto from_idx =
                patch_layout_t::padded_layout_t::linear_index(from_idxs);
            intergrid_operator_t::interpolation(
                self_patch, to_idx, child_offset, other_patch, from_idx
            );
        }
    };

    inline static constexpr boundary_t boundary{};
    inline static constexpr same_t     same{};
    inline static constexpr finer_t    finer{};
    inline static constexpr coarser_t  coarser{};
};

} // namespace patches

} // namespace amr::ndt::utils

#endif // AMR_INCLUDED_PATCH_UTILS
