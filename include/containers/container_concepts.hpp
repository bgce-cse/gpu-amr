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

template <typename I>
concept MultiIndex = requires() {
    { I::elements() } -> std::same_as<typename I::size_type>;
    { I::rank() } -> std::same_as<typename I::rank_t>;
} && std::ranges::range<I>;

template <typename L>
concept StaticLayout = requires(
    const typename L::index_t       i,
    const typename L::multi_index_t midx,
    const typename L::index_t (&idxs)[L::rank()]
) {
    typename L::size_type;
    typename L::index_t;
    typename L::rank_t;
    typename L::multi_index_t;
    { L::rank() } -> std::same_as<typename L::rank_t>;
    { L::elements() } -> std::same_as<typename L::size_type>;
    { L::flat_size() } -> std::same_as<typename L::size_type>;
    L::sizes();
    { L::size(i) } -> std::same_as<typename L::size_type>;
    L::strides();
    { L::stride(i) } -> std::same_as<typename L::size_type>;
    { L::linear_index(midx) } -> std::same_as<typename L::index_t>;
    { L::linear_index(idxs) } -> std::same_as<typename L::index_t>;
};

template <typename S>
concept StaticShape = requires() {
    typename S::size_type;
    typename S::rank_t;
    { S::rank() } -> std::same_as<typename S::rank_t>;
    S::sizes();
    S::elements();
};

template <typename A>
concept StaticMDArray = requires(A a, typename A::size_type s, typename A::index_t i) {
    { A::rank() } -> std::same_as<typename A::rank_t>;
};

} // namespace amr::containers::concepts

#endif // AMR_INCLUDED_CONTAINER_CONCEPTS
