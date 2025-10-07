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

namespace tensor
{

namespace detail
{
template <typename T, std::integral auto H, std::integral auto Size, std::size_t Rank, std::size_t... Is>
constexpr auto make_hypercube_type_impl(std::index_sequence<Is...>)
    -> static_tensor<T, H, ((void)Is, Size)...>  // ← Use the halo parameter H
{
    return {};
}

} // namespace detail

template <typename T, std::integral auto H, std::integral auto Size, std::size_t Rank>
struct hypercube
{
    using type = decltype(detail::make_hypercube_type_impl<T, H, Size, Rank>(
        std::make_index_sequence<Rank>{}
    ));
};

template <typename T, std::integral auto H, std::integral auto Size, std::size_t Rank>
using hypercube_t = typename hypercube<T, H, Size, Rank>::type;

} // namespace tensor

} // namespace types

} // namespace amr::containers::utils

#endif // AMR_INCLUDED_CONTAINER_UTILS
