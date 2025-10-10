#ifndef AMR_INCLUDED_CONTAINER_UTILS
#define AMR_INCLUDED_CONTAINER_UTILS

#include "static_tensor.hpp"
#include <concepts>
#include <type_traits>
#include <utility>

namespace amr::containers::utils
{

namespace types
{

namespace layout
{

template <typename T>
struct padded_layout;

template <std::integral auto N, std::integral auto... Ns>
struct padded_layout<static_layout<N, Ns...>>
{
    using size_type = std::common_type_t<decltype(N), decltype(Ns)...>;
    template <std::integral auto Pad>
    using type = static_layout<
        static_cast<size_type>(N + Pad),
        static_cast<size_type>(Ns + Pad)...>;
};

} // namespace layout

namespace tensor
{

namespace detail
{
template <typename T, std::integral auto Size, std::size_t Rank, std::size_t... Is>
constexpr auto make_hypercube_type_impl(std::index_sequence<Is...>)
    -> static_tensor<T, static_layout<((void)Is, Size)...>>
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

} // namespace tensor

} // namespace types

} // namespace amr::containers::utils

#endif // AMR_INCLUDED_CONTAINER_UTILS
