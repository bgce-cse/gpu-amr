#ifndef AMR_INCLUDED_NDUTILS
#define AMR_INCLUDED_NDUTILS

#include "containers/container_concepts.hpp"
#include "ndconcepts.hpp"
#include "utility/constexpr_functions.hpp"
#include <algorithm>
#include <cassert>
#include <concepts>
#include <type_traits>

namespace amr::ndt::utils
{

[[nodiscard]]
consteval auto subdivisions(
    std::unsigned_integral auto dim,
    std::unsigned_integral auto fanout
) noexcept -> std::common_type_t<decltype(dim), decltype(fanout)>
{
    using return_t = std::common_type_t<decltype(dim), decltype(fanout)>;
    return utility::cx_functions::pow(return_t{ dim }, return_t{ fanout });
}

namespace patches
{

[[nodiscard]]
consteval auto multiples_of(
    std::ranges::input_range auto const& r,
    std::integral auto const             x
) noexcept -> bool
    requires std::is_same_v<
        std::ranges::range_value_t<std::remove_cvref_t<decltype(r)>>,
        std::remove_const_t<decltype(x)>>
{
    return std::ranges::all_of(r, [x](auto const& e) { return e % x == 0; });
}

template <containers::concepts::StaticLayout Layout>
[[nodiscard]]
constexpr auto is_halo_cell(
    typename Layout::index_t   linear_index,
    typename Layout::size_type halo_width
) noexcept -> bool
{
    using layout_t                       = Layout;
    using size_type                      = layout_t::size_type;
    static constexpr auto        rank    = layout_t::rank();
    static constexpr auto const& strides = layout_t::strides();
    static constexpr auto const& sizes   = layout_t::sizes();

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

template <concepts::PatchType Patch>
[[nodiscard]]
consteval auto fragmentation_patch_maps() noexcept -> containers::utils::types::tensor::
    hypercube_t<Patch, Patch::fanout(), Patch::dimension()>
{
    using patch_t                = Patch;
    static constexpr auto fanout = patch_t::fanout();

    using layout_t = typename patch_t::padded_layout_t;
    using index_t  = typename patch_t::index_t;
    using tensor_t = containers::static_tensor<index_t, layout_t>;
    using patch_shape_t =
        containers::utils::types::tensor::hypercube_t<patch_t, fanout, tensor_t::rank()>;
    patch_shape_t to{};

    auto                   idx        = typename tensor_t::multi_index_t{};
    static constexpr auto& strides    = layout_t::strides();
    static constexpr auto& data_shape = patch_t::data_layout_t::sizes();
    do
    {
        static constexpr index_t halo_width = patch_t::halo_width();
        const auto               linear_idx = layout_t::linear_index(idx);
        const auto               is_halo = is_halo_cell<layout_t>(linear_idx, halo_width);
        const auto               offset  = is_halo ? index_t{}
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
        auto                     out_patch_idx = typename patch_shape_t::multi_index_t{};
        do
        {
            const auto linear_out_idx      = patch_shape_t::linear_index(out_patch_idx);
            const auto base                = is_halo ? index_t{ -(linear_out_idx + 1) }
                                                     : [&out_patch_idx]()
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

} // namespace patches

} // namespace amr::ndt::utils

#endif // AMR_INCLUDED_NDUTILS
