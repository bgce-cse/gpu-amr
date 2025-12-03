#ifndef AMR_INCLUDED_PATCH_UTILS
#define AMR_INCLUDED_PATCH_UTILS

#include "containers/container_manipulations.hpp"
#include "containers/container_utils.hpp"
#include "containers/static_tensor.hpp"
#include "ndconcepts.hpp"
#include "ndutils.hpp"
#include "utility/compile_time_utility.hpp"
#include "neighbor.hpp"
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

    assert(linear_index < layout_t::flat_size());
    if (std::is_signed_v<size_type>)
    {
        assert(linear_index >= 0);
    }

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
                    neighbor.dim_offset,
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

    static constexpr auto s_halo_width = patch_layout_t::halo_width();
    static constexpr auto s_dimension  = patch_layout_t::rank();
    static constexpr auto s_1d_fanout  = patch_index_t::fanout();
    static constexpr auto s_nd_fanout  = patch_index_t::nd_fanout();
    static constexpr auto s_sizes      = patch_layout_t::data_layout_t::sizes();

    struct boundary_t
    {
        static constexpr auto operator()(
            [[maybe_unused]] auto const& p_i,
            [[maybe_unused]] auto const& direction,
            [[maybe_unused]] auto&&... args
        ) noexcept -> void
        {
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
            using direction_t   = std::remove_cvref_t<decltype(direction)>;
            const auto dim      = direction.dimension();
            const auto positive = direction_t::is_positive(direction);
            [[assume(idxs[dim] >= s_halo_width)]];
            auto from_idxs = idxs;
            from_idxs[dim] +=
                positive ? -index_t{ s_sizes[dim] } : index_t{ s_sizes[dim] };
            self_patch[idxs] = other_patch[from_idxs];
        }
    };

    struct finer_t
    {
        static constexpr auto operator()(
            auto&                                              current_patch,
            std::ranges::contiguous_range auto const&          neighbor_patches,
            auto const&                                        direction,
            auto const&                                        idxs,
            [[maybe_unused]] auto&&... args
        ) noexcept -> void
        {
            using direction_t = std::remove_cvref_t<decltype(direction)>;
            using value_t = std::remove_cvref_t<decltype(current_patch[idxs])>;
            
            const auto dim      = direction.dimension();
            const auto positive = direction_t::is_positive(direction);
            
            std::printf("=== finer_t operator ===\n");
            std::printf("Direction: dim=%zu, positive=%d\n", 
                        static_cast<std::size_t>(dim), static_cast<int>(positive));
            std::printf("Input idxs: [");
            for (index_t d = 0; d < s_dimension; ++d)
                std::printf("%zu%s", static_cast<std::size_t>(idxs[d]), 
                           d < s_dimension - 1 ? ", " : "");
            std::printf("]\n");
            
            // Step 1: Determine which fine patch to read from based on perpendicular coordinates
            auto compute_fine_patch_index = [&]() -> index_t
            {
                index_t patch_linear_idx = 0;
                index_t stride = 1;
                
                std::printf("Computing fine patch index:\n");
                for (index_t d = 0; d < s_dimension; ++d)
                {
                    if(d == dim){
                        continue;
                    }
                    const auto section_size = s_sizes[d] / s_1d_fanout;
                    const auto section_idx = (idxs[d] - s_halo_width) / section_size;
                    
                    std::printf("  dim %zu: section_size=%zu, section_idx=%zu, stride=%zu\n",
                               static_cast<std::size_t>(d),
                               static_cast<std::size_t>(section_size),
                               static_cast<std::size_t>(section_idx),
                               static_cast<std::size_t>(stride));
                    
                    patch_linear_idx += section_idx * stride;
                    stride *= s_1d_fanout;
                }
                return patch_linear_idx;
            };
            
            const auto fine_patch_idx = compute_fine_patch_index();
            std::printf("Selected fine_patch_idx: %zu\n", static_cast<std::size_t>(fine_patch_idx));
            
            auto& fine_patch = neighbor_patches[fine_patch_idx].get();
            
            // Step 2: Compute the base "parent" position in the fine patch (same as same_t)
            auto from_idxs = idxs;
            from_idxs[dim] += positive ? -index_t{s_sizes[dim]} : index_t{s_sizes[dim]};
            
            std::printf("from_idxs (after offset): [");
            for (index_t d = 0; d < s_dimension; ++d)
                std::printf("%zu%s", static_cast<std::size_t>(from_idxs[d]), 
                           d < s_dimension - 1 ? ", " : "");
            std::printf("]\n");
            
            // Step 3: Compute fanout^dimension child indices relative to from_idxs
            auto base_fine_idxs = from_idxs;
            for (index_t d = 0; d < s_dimension; ++d)
            {
                base_fine_idxs[d] = ((from_idxs[d] - s_halo_width) * s_1d_fanout) % s_sizes[d] + s_halo_width;

                std::printf("  base_fine_idxs[%zu]: (%zu - %zu) * %zu + %zu = %zu\n",
                           static_cast<std::size_t>(d),
                           static_cast<std::size_t>(from_idxs[d]),
                           static_cast<std::size_t>(s_halo_width),
                           static_cast<std::size_t>(s_1d_fanout),
                           static_cast<std::size_t>(s_halo_width),
                           static_cast<std::size_t>(base_fine_idxs[d]));
            }
            
            std::printf("base_fine_idxs: [");
            for (index_t d = 0; d < s_dimension; ++d)
                std::printf("%zu%s", static_cast<std::size_t>(base_fine_idxs[d]), 
                           d < s_dimension - 1 ? ", " : "");
            std::printf("]\n");
            
            // Step 4: Average over all fanout^dimension fine cells
            value_t sum{};
            constexpr index_t num_fine_cells = []() constexpr {
                index_t count = 1;
                for (index_t d = 0; d < s_dimension; ++d)
                    count *= s_1d_fanout;
                return count;
            }();
            
            std::printf("Averaging over %zu fine cells:\n", static_cast<std::size_t>(num_fine_cells));
            
            for (index_t fine_offset = 0; fine_offset < num_fine_cells; ++fine_offset)
            {
                auto fine_cell_idxs = base_fine_idxs;
                index_t remaining = fine_offset;
                for (index_t d = 0; d < s_dimension; ++d)
                {
                    fine_cell_idxs[d] += remaining % s_1d_fanout;
                    remaining /= s_1d_fanout;
                }
                
                const auto cell_value = fine_patch[fine_cell_idxs];
                sum += cell_value;
                
                std::printf("  fine_offset=%zu, fine_cell_idxs=[", 
                           static_cast<std::size_t>(fine_offset));
                for (index_t d = 0; d < s_dimension; ++d)
                    std::printf("%zu%s", static_cast<std::size_t>(fine_cell_idxs[d]), 
                               d < s_dimension - 1 ? ", " : "");
                std::printf("], value=");
                
                // Print value based on type (this might need adjustment for your types)
                if constexpr (std::is_arithmetic_v<value_t>)
                    std::printf("%g\n", static_cast<double>(cell_value));
                else
                    std::printf("<complex_type>\n");
            }
            
            const auto averaged_value = sum / static_cast<value_t>(num_fine_cells);
            std::printf("Sum: ");
            if constexpr (std::is_arithmetic_v<value_t>)
                std::printf("%g", static_cast<double>(sum));
            else
                std::printf("<complex_type>");
            std::printf(", Averaged value: ");
            if constexpr (std::is_arithmetic_v<value_t>)
                std::printf("%g", static_cast<double>(averaged_value));
            else
                std::printf("<complex_type>");
            std::printf("\n");
            
            current_patch[idxs] = averaged_value;
            std::printf("=== finer_t done ===\n\n");
        }
    };

    struct coarser_t
    {
        static constexpr auto operator()(
            auto&       self_patch,
            auto const& other_patch,
            auto const& direction,
            auto const& dim_offset,      // Just a plain unsigned int!
            auto const& idxs,
            [[maybe_unused]] auto&&... args
        ) noexcept -> void
        {
            using direction_t = std::remove_cvref_t<decltype(direction)>;
            using value_t = std::remove_cvref_t<decltype(self_patch[idxs])>;
            
            const auto dim      = direction.dimension();
            const auto positive = direction_t::is_positive(direction);
            
            std::printf("=== coarser_t operator ===\n");
            std::printf("Direction: dim=%zu, positive=%d\n", 
                        static_cast<std::size_t>(dim), static_cast<int>(positive));
            std::printf("Input idxs (fine halo): [");
            for (index_t d = 0; d < s_dimension; ++d)
                std::printf("%zu%s", static_cast<std::size_t>(idxs[d]), 
                           d < s_dimension - 1 ? ", " : "");
            std::printf("]\n");
            std::printf("dim_offset=%u\n", static_cast<unsigned int>(dim_offset));
            
            // Step 1: Compute from_idxs
            auto from_idxs = idxs;
            from_idxs[dim] += positive ? -index_t{s_sizes[dim]} : index_t{s_sizes[dim]};
            
            std::printf("from_idxs (fine space, after offset): [");
            for (index_t d = 0; d < s_dimension; ++d)
                std::printf("%zu%s", static_cast<std::size_t>(from_idxs[d]), 
                           d < s_dimension - 1 ? ", " : "");
            std::printf("]\n");
            
            // Step 2: Map fine cell to coarse cell
            auto coarse_idxs = from_idxs;
            for (index_t d = 0; d < s_dimension; ++d)
            {
                const auto fine_coord = from_idxs[d] - s_halo_width;
                const auto local_fine_coord = fine_coord % s_sizes[d];
                const auto coarse_coord = local_fine_coord / s_1d_fanout;
                
                // In the exchange dimension, use dim_offset
                // In other dimensions, compute from current position
                index_t child_base_offset;
                if (d == dim)
                {
                    // Use the provided dim_offset for the exchange dimension
                    child_base_offset =  positive? 0 : (s_sizes[d] / s_1d_fanout) ;
                }
                else
                {
                    // Compute which section we're in for perpendicular dimensions
                    // const auto section_idx = (idxs[d] - s_halo_width) / (s_sizes[d] / s_1d_fanout);
                    child_base_offset = dim_offset * (s_sizes[d] / s_1d_fanout);
                }
                
                coarse_idxs[d] = coarse_coord + child_base_offset + s_halo_width;
                
                std::printf("  coarse_idxs[%zu]: fine_coord=%zu, local=%zu, coarse=%zu, "
                           "child_offset=%zu, final=%zu\n",
                           static_cast<std::size_t>(d),
                           static_cast<std::size_t>(fine_coord),
                           static_cast<std::size_t>(local_fine_coord),
                           static_cast<std::size_t>(coarse_coord),
                           static_cast<std::size_t>(child_base_offset),
                           static_cast<std::size_t>(coarse_idxs[d]));
            }
            
            std::printf("coarse_idxs (coarse patch): [");
            for (index_t d = 0; d < s_dimension; ++d)
                std::printf("%zu%s", static_cast<std::size_t>(coarse_idxs[d]), 
                           d < s_dimension - 1 ? ", " : "");
            std::printf("]\n");
            
            const auto coarse_value = other_patch[coarse_idxs];
            self_patch[idxs] = coarse_value;
            
            std::printf("Copied value: ");
            if constexpr (std::is_arithmetic_v<value_t>)
                std::printf("%g", static_cast<double>(coarse_value));
            else
                std::printf("<complex_type>");
            std::printf("\n=== coarser_t done ===\n\n");
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
