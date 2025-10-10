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
concept Vector = requires(V v, typename V::size_type i) { v[i]; } && Container<V> &&
                 std::is_trivially_constructible_v<V>;

template <typename L>
concept StaticLayout = requires(
    const typename L::multi_index_t midx,
    const typename L::index_t (&idxs)[L::s_rank]
) {
    typename L::size_type;
    typename L::index_t;
    typename L::rank_t;
    typename L::multi_index_t;
    L::s_rank;
    L::s_sizes;
    L::s_strides;
    L::s_flat_size;
    { L::linear_index(midx) } -> std::same_as<typename L::index_t>;
    { L::linear_index(idxs) } -> std::same_as<typename L::index_t>;
};

template <typename L>
concept StaticShape = requires() {
    typename L::size_type;
    typename L::rank_t;
    L::s_rank;
    L::s_sizes;
    L::s_elements;
};

} // namespace amr::containers::concepts

#endif // AMR_INCLUDED_CONTAINER_CONCEPTS
