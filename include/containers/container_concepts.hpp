#ifndef AMR_INCLUDED_CONTAINER_CONCEPTS
#define AMR_INCLUDED_CONTAINER_CONCEPTS

#include <ranges>
#include <type_traits>

namespace amr::containers::concepts
{

template <typename C>
concept Container = requires(C c) {
    typename C::size_type;
    typename C::value_type;
} && std::ranges::sized_range<C>;

template <typename V>
concept Vector = requires(V v, typename V::size_type i) {
    v[i];
} && Container<V> && std::is_trivially_constructible_v<V>;

} // namespace amr::containers::concepts

#endif // AMR_INCLUDED_CONTAINER_CONCEPTS
