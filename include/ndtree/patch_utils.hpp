#ifndef AMR_INCLUDED_PATCH_UTILS
#define AMR_INCLUDED_PATCH_UTILS

#include "containers/container_manipulations.hpp"
#include "containers/container_utils.hpp"
#include "containers/static_tensor.hpp"
#include "containers/static_vector.hpp"
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
    using patch_layout_t                    = Layout;
    using layout_t                          = typename patch_layout_t::padded_layout_t;
    using size_type                         = layout_t::size_type;
    static constexpr auto        rank       = layout_t::rank();
    static constexpr auto const& strides    = layout_t::strides();
    static constexpr auto const& sizes      = layout_t::sizes();
    static constexpr size_type   halo_width = patch_layout_t::halo_width();

    utility::contracts::assert_index(linear_index, layout_t::flat_size());

    for (auto j = decltype(rank){}; j != rank; ++j)
    {
        const auto relative_idx = static_cast<size_type>(linear_index / strides[j]);
        assert(relative_idx < sizes[j]);
        if (relative_idx < halo_width || relative_idx >= sizes[j] - halo_width)
        {
            return true;
        }
        linear_index %= strides[j];
    }
    return false;
}

template <concepts::PatchLayout Patch_Layout, std::integral auto Fanout>
[[nodiscard]]
consteval auto fragmentation_patch_maps() noexcept
    -> containers::utils::types::tensor::hypercube_t<
        containers::static_tensor<
            typename Patch_Layout::index_t,
            typename Patch_Layout::padded_layout_t>,
        Fanout,
        Patch_Layout::rank()>
{
    using patch_layout_t         = Patch_Layout;
    static constexpr auto fanout = static_cast<typename patch_layout_t::index_t>(Fanout);

    using layout_t = typename patch_layout_t::padded_layout_t;
    using index_t  = typename patch_layout_t::index_t;
    using tensor_t = typename containers::static_tensor<index_t, layout_t>;
    using patch_shape_t =
        containers::utils::types::tensor::hypercube_t<tensor_t, fanout, tensor_t::rank()>;
    patch_shape_t to{};

    static constexpr auto& strides = layout_t::strides();
    // static constexpr auto& patch_shape = layout_t::sizes();
    static constexpr auto& data_shape = patch_layout_t::data_layout_t::sizes();
    auto                   idx        = typename tensor_t::multi_index_t{};
    do
    {
        static constexpr index_t halo_width = patch_layout_t::halo_width();
        const auto               linear_idx = layout_t::linear_index(idx);
        const auto               is_halo    = is_halo_cell<patch_layout_t>(linear_idx);
        const auto               offset        = is_halo 
            ? [&idx]()
            {
                index_t ret{};
                for(index_t i = 0; i != layout_t::rank(); ++i)
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
                    assert(i >= halo_width);
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
                    assert(data_shape[i] % fanout == 0);
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
            [&tree, &p_i, &n_idx, &args...](
                typename n_linear_idx_varant_t::finer const& neighbor
            )
            {
                using neighbor_t = std::remove_cvref_t<decltype(neighbor)>;
                using patch_t =
                    typename std::remove_cvref_t<decltype(tree)>::template patch_t<T>;

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

template <concepts::PatchIndex Patch_Index, concepts::PatchLayout Patch_Layout>
struct halo_exchange_impl_t
{
    using patch_layout_t = Patch_Layout;
    using patch_index_t  = Patch_Index;
    using index_t        = typename patch_layout_t::index_t;

    // TODO: Revisit this after propperly typing patch_index_t
    static constexpr auto s_halo_width =
        static_cast<index_t>(patch_layout_t::halo_width());
    static constexpr auto s_dimension = static_cast<index_t>(patch_layout_t::rank());
    static constexpr auto s_1d_fanout = static_cast<index_t>(patch_index_t::fanout());
    static constexpr auto s_nd_fanout = static_cast<index_t>(patch_index_t::nd_fanout());
    static constexpr auto s_sizes     = patch_layout_t::data_layout_t::sizes();

    struct boundary_t
    {
        static constexpr auto operator()(
            [[maybe_unused]] auto const& p_i,
            [[maybe_unused]] auto const& direction,
            [[maybe_unused]] auto&&... args
        ) noexcept -> void
        {
            DEFAULT_SOURCE_LOG_TRACE(
                std::string("Boundary halo exchange in direction ") +
                std::string(direction.repr())
            );
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
            DEFAULT_SOURCE_LOG_TRACE(
                std::string("Same halo exchange in direction ") +
                std::string(direction.repr())
            );
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
            auto const&                               direction,
            auto const&                               idxs,
            [[maybe_unused]] auto&&... args
        ) noexcept -> void
        {
            DEFAULT_SOURCE_LOG_TRACE(
                std::string("Finer halo exchange in direction ") +
                std::string(direction.repr())
            );
            using value_t = std::remove_cvref_t<decltype(current_patch[idxs])>;

            const auto dim      = direction.dimension();
            const auto positive = direction.is_positive();

            const auto compute_fine_patch_index = [&idxs, &dim]() -> index_t
            {
                index_t patch_linear_idx = 0;
                index_t stride           = 1;

                for (index_t d = 0; d < s_dimension; ++d)
                {
                    if (d == dim)
                    {
                        continue;
                    }
                    const auto section_size = s_sizes[d] / s_1d_fanout;
                    const auto section_idx  = (idxs[d] - s_halo_width) / section_size;

                    patch_linear_idx += section_idx * stride;
                    stride *= s_1d_fanout;
                }
                return patch_linear_idx;
            };

            const auto fine_patch_idx = compute_fine_patch_index();
            auto&      fine_patch     = neighbor_patches[fine_patch_idx].get();

            auto from_idxs = idxs;
            from_idxs[dim] = positive ? (from_idxs[dim] - index_t{ s_sizes[dim] })
                                      : (from_idxs[dim] + index_t{ s_sizes[dim] });

            auto base_fine_idxs = from_idxs;
            for (index_t d = 0; d < s_dimension; ++d)
            {
                base_fine_idxs[d] =
                    ((from_idxs[d] - s_halo_width) * s_1d_fanout) % s_sizes[d] +
                    s_halo_width;
            }

            value_t sum{};
            for (index_t fine_offset = 0; fine_offset < s_nd_fanout; ++fine_offset)
            {
                auto    fine_cell_idxs = base_fine_idxs;
                index_t remaining      = fine_offset;
                for (index_t d = 0; d < s_dimension; ++d)
                {
                    fine_cell_idxs[d] += remaining % s_1d_fanout;
                    remaining /= s_1d_fanout;
                }
                sum += fine_patch[fine_cell_idxs];
            }

            current_patch[idxs] = sum / static_cast<value_t>(s_nd_fanout);
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
            DEFAULT_SOURCE_LOG_TRACE(
                std::string("Coarser halo exchange in direction ") +
                std::string(direction.repr())
            );
            // std::cout << "\nDSizes:\t";
            // for (auto const& e : s_sizes)
            //     std::cout << e << ' ';
            // std::cout << "\nIdxs:\t";
            // for (auto const& e : idxs)
            //     std::cout << e << ' ';
            // std::cout << "\nQ:\t";
            // for (auto const& e : contact_quadrant)
            //     std::cout << e << ' ';
            // std::cout << '\n';

            const auto dim      = direction.dimension();
            const auto positive = direction.is_positive();

            containers::static_vector<index_t, s_dimension> from_idxs;

            for (auto i = index_t{}; i != s_dimension; ++i)
            {
                assert(
                    i == dim
                        ? (idxs[i] < s_1d_fanout || idxs[i] >= s_1d_fanout + s_sizes[i])
                        : (idxs[i] >= s_1d_fanout && idxs[i] < s_1d_fanout + s_sizes[i])
                );
                const auto cells_per_block = (s_sizes[i] / s_1d_fanout);
                const auto block_offset =
                    (i == dim && positive)
                        ? index_t{}
                        : (cells_per_block - s_halo_width / s_1d_fanout);
                const auto idx_offset =
                    i == dim ? (direction.is_positive() ? s_sizes[i] + s_halo_width
                                                        : index_t{})
                             : s_halo_width;
                assert(idxs[i] >= idx_offset);
                utility::contracts::assert_index(contact_quadrant[i], s_1d_fanout);
                // std::cout << (quadrant[i] * cells_per_block) << '\n';
                from_idxs[i] = s_halo_width + (contact_quadrant[i] * cells_per_block) +
                               block_offset + (idxs[i] - idx_offset) / s_1d_fanout;
                // std::cout << "hw:\t" << s_halo_width << '\n';
                // std::cout << "dim:\t" << direction.dimension() << '\n';
                // std::cout << "cpb:\t" << cells_per_block << '\n';
                // std::cout << "do:\t" << dim_offset << '\n';
                // std::cout << "o:\t" << offset << '\n';
                // std::cout << "1df:\t" << s_1d_fanout << '\n';
                // std::cout << "From:\t";
                // for (auto const& e : coarse_cell_coords)
                //     std::cout << e << ' ';
                // std::cout << '\n';
                utility::contracts::assert_index(from_idxs[i] - s_halo_width, s_sizes[i]);
            }

            // std::cout << "\nFrom:\t";
            // for (auto const& e : from_idxs)
            //     std::cout << e << ' ';
            // std::cout << '\n';

            self_patch[idxs] = other_patch[from_idxs];
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
