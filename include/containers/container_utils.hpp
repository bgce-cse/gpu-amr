#ifndef AMR_INCLUDED_CONTAINER_UTILS
#define AMR_INCLUDED_CONTAINER_UTILS

#include "static_layout.hpp"
#include "static_shape.hpp"
#include "static_tensor.hpp"
#include <concepts>
#include <utility>

namespace amr::containers::utils
{

namespace types
{

namespace sequences
{

template <typename, typename>
struct concatenate;

template <std::integral I, I... As, I... Bs>
struct concatenate<std::integer_sequence<I, As...>, std::integer_sequence<I, Bs...>>
{
    using type = std::integer_sequence<I, As..., Bs...>;
};

template <typename A, typename B>
using concatenate_t = typename concatenate<A, B>::type;

} // namespace sequences

namespace shape
{

template <typename>
struct static_shape_wrapper;

template <std::integral I, I... Ns>
struct static_shape_wrapper<std::integer_sequence<I, Ns...>>
{
    using type = static_shape<Ns...>;
};

template <typename T>
using static_shape_wrapper_t = typename static_shape_wrapper<T>::type;

} // namespace shape

namespace layout
{

template <typename T>
struct padded_layout;

template <std::integral auto N, std::integral auto... Ns>
struct padded_layout<static_layout<static_shape<N, Ns...>>>
{
    using size_type = typename static_layout<static_shape<N, Ns...>>::size_type;
    template <std::integral auto Pad>
    using type = static_layout<static_shape<
        static_cast<size_type>(N + Pad),
        static_cast<size_type>(Ns + Pad)...>>;
};

} // namespace layout

namespace tensor
{

namespace detail
{

template <typename T, std::integral auto Size, std::size_t Rank, std::size_t... Is>
constexpr auto make_hypercube_type_impl(std::index_sequence<Is...>)
    -> static_tensor<T, static_layout<static_shape<((void)Is, Size)...>>>
{
    return {};
}

} // namespace detail

template <typename T, std::integral auto Size, std::size_t Rank>
struct hypercube
{
    using type = decltype(detail::make_hypercube_type_impl<T, Size, Rank>(
        std::make_index_sequence<Rank>{}
    ));
};

template <typename T, std::integral auto Size, std::size_t Rank>
using hypercube_t = typename hypercube<T, Size, Rank>::type;

template <concepts::StaticContainer T1, concepts::StaticContainer T2>
struct tensor_product_result
{
    using value_type =
        std::common_type_t<typename T1::value_type, typename T2::value_type>;
    using sizes_pack_t =
        sequences::concatenate_t<typename T1::size_pack_t, typename T2::size_pack_t>;
    using type = static_tensor<
        value_type,
        static_layout<utils::types::shape::static_shape_wrapper_t<sizes_pack_t>>>;
};

template <concepts::StaticContainer T1, concepts::StaticContainer T2>
using tensor_product_result_t = typename tensor_product_result<T1, T2>::type;

} // namespace tensor

} // namespace types

} // namespace amr::containers::utils

#endif // AMR_INCLUDED_CONTAINER_UTILS
