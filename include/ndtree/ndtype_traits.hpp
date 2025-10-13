#ifndef AMR_INCLUDED_TYPE_TRAITS
#define AMR_INCLUDED_TYPE_TRAITS

#include "ndconcepts.hpp"
#include <tuple>

namespace amr::ndt::type_traits
{
namespace detail
{

template <template <typename T> typename Modifier, typename>
struct tuple_type_apply_impl;

template <template <typename T> typename Modifier, typename... Ts>
struct tuple_type_apply_impl<Modifier, std::tuple<Ts...>>
{
    using type = std::tuple<Modifier<Ts>...>;
};

} // namespace detail

template <template <typename T> typename Modifier, typename Tuple>
using tuple_type_apply_t = typename detail::tuple_type_apply_impl<Modifier, Tuple>::type;

} // namespace amr::ndt::type_traits

#endif // AMR_INCLUDED_TYPE_TRAITS
