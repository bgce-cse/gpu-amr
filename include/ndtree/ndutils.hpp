#ifndef AMR_INCLUDED_NDUTILS
#define AMR_INCLUDED_NDUTILS

#include "containers/container_utils.hpp"
#include "containers/static_layout.hpp"
#include "containers/static_tensor.hpp"
#include "ndconcepts.hpp"
#include "utility/constexpr_functions.hpp"
#include <algorithm>
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
        decltype(x)>
{
    return std::ranges::all_of(r, [x](auto const& e) { return e % x == 0; });
}

template <
    std::integral      Index_Type,
    std::integral auto Fanout,
    std::integral auto N,
    std::integral auto... Ns>
[[nodiscard]]
consteval auto fragmentation_patch_maps(containers::static_layout<N, Ns...>) noexcept
    -> containers::utils::types::tensor::hypercube_t<
        containers::static_tensor<Index_Type, N, Ns...>,
        Fanout,
        containers::static_tensor<Index_Type, N, Ns...>::s_rank>
{
    using index_t  = Index_Type;
    using tensor_t = containers::static_tensor<index_t, N, Ns...>;
    using patch_shape_t =
        containers::utils::types::tensor::hypercube_t<tensor_t, Fanout, tensor_t::s_rank>;
    patch_shape_t to{};

    auto idx           = typename tensor_t::multi_index_t{};
    auto sized_strides = decltype(tensor_t::s_strides){};
    std::transform(
        std::cbegin(tensor_t::s_strides),
        std::cend(tensor_t::s_strides),
        std::cbegin(tensor_t::s_sizes),
        std::begin(sized_strides),
        std::multiplies{}
    );
    do
    {
        const auto offset = std::transform_reduce(
            std::cbegin(idx),
            std::cend(idx),
            std::cbegin(tensor_t::s_strides),
            index_t{},
            std::plus{},
            [](index_t const i, index_t const s) { return (i / Fanout) * s; }
        );
        auto out_patch_idx = typename patch_shape_t::multi_index_t{};
        do
        {
            const auto base = std::transform_reduce(
                                  std::cbegin(out_patch_idx),
                                  std::cend(out_patch_idx),
                                  std::cbegin(sized_strides),
                                  index_t{}
                              ) /
                              Fanout;
            to[out_patch_idx][idx] = offset + base;
        } while (out_patch_idx.increment());
    } while (idx.increment());
    return to;
}

} // namespace patches

} // namespace amr::ndt::utils

#endif // AMR_INCLUDED_NDUTILS
