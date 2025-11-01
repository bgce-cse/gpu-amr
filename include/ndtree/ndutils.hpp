#ifndef AMR_INCLUDED_NDUTILS
#define AMR_INCLUDED_NDUTILS

#include "containers/container_concepts.hpp"
#include "containers/container_manipulations.hpp"
#include "containers/container_utils.hpp"
#include "containers/static_tensor.hpp"
#include "ndconcepts.hpp"
#include "utility/constexpr_functions.hpp"
#include <algorithm>
#include <cassert>
#include <concepts>
#include <type_traits>

namespace amr::ndt::utils
{

[[nodiscard]]
consteval auto
    compute_nd_fanout(std::integral auto dim, std::integral auto fanout) noexcept
    -> std::common_type_t<decltype(dim), decltype(fanout)>
{
    using return_t = std::common_type_t<decltype(dim), decltype(fanout)>;
    return utility::cx_functions::pow(return_t{ dim }, return_t{ fanout });
}

template <class... Ts>
struct overloads : Ts...
{
    using Ts::operator()...;
};

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
        const size_type relative_idx = linear_index / strides[j];
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

template <concepts::Direction auto D, concepts::Patch Patch>
constexpr auto halo_apply_unroll_impl(Patch&& p, auto&& fn, auto&&... args) noexcept
    -> void
{
    if constexpr (D == decltype(D)::sentinel())
    {
    }
    else
    {
        containers::manipulators::for_each<
            typename Patch::template halo_iteration_control_t<D>>(
            std::forward<decltype(p)>(p).data(),
            std::forward<decltype(fn)>(fn),
            std::forward<decltype(args)>(args)...
        );
        halo_apply_unroll_impl<decltype(D)::advance(D)>(
            std::forward<decltype(p)>(p),
            std::forward<decltype(fn)>(fn),
            std::forward<decltype(args)>(args)...
        );
    }
}

} // namespace detail

template <concepts::Direction D_Type, concepts::Patch Patch>
[[nodiscard]]
auto halo_apply(Patch&& p, auto&& fn, auto&&... args)
{
    detail::halo_apply_unroll_impl<D_Type::first()>(
        std::forward<decltype(p)>(p),
        std::forward<decltype(fn)>(fn),
        std::forward<decltype(args)>(args)...
    );
}

} // namespace patches

} // namespace amr::ndt::utils

#endif // AMR_INCLUDED_NDUTILS
