#ifndef AMR_INCLUDED_NDCONCEPTS
#define AMR_INCLUDED_NDCONCEPTS

#include <concepts>
#include <functional>
#include <iterator>
#include <optional>
#include <string>
#include <type_traits>

namespace amr::ndt::concepts
{

template <typename T>
concept TypeMap = requires {
    typename T::type;
    { T::index() } -> std::same_as<std::size_t>;
};

namespace detail
{

template <typename>
constexpr bool type_map_tuple_impl = false;
template <template <class...> class Tuple, class... Types>
    requires(TypeMap<Types> && ...)
constexpr bool type_map_tuple_impl<Tuple<Types...>> = true;

} // namespace detail

template <typename T>
concept DeconstructibleType = requires {
    typename T::deconstructed_types_map_t;
} && detail::type_map_tuple_impl<typename T::deconstructed_types_map_t>;

template <typename I>
concept PatchIndex =
    requires(
        const I                       i,
        const typename I::direction_t d,
        const typename I::offset_t    offset
    ) {
        { I::rank() } -> std::same_as<typename I::size_type>;
        { I::fanout() } -> std::same_as<typename I::size_type>;
        { I::nd_fanout() } -> std::same_as<typename I::size_type>;
        { I::max_depth() } -> std::integral;
        { I::root() } -> std::same_as<I>;
        { I::parent_of(i) } -> std::same_as<I>;
        { I::child_of(i, offset) } -> std::same_as<I>;
        { I::neighbor_at(i, d) } -> std::same_as<std::optional<I>>;
        { I::offset_of(i) } -> std::same_as<typename I::offset_t>;
        { I::offset(i, offset) } -> std::same_as<I>; // TODO: Rethink
        { I::level(i) } -> std::same_as<typename I::level_t>;
        { i.id() } -> std::same_as<typename I::mask_t>;
        { i.repr() } -> std::same_as<std::string>;
        { std::less{}(i, i) } -> std::convertible_to<bool>;
    } &&
    std::integral<typename I::size_type> && std::unsigned_integral<typename I::mask_t> &&
    std::equality_comparable<I>;

template <typename P>
concept Patch = requires(
    P const                                cp,
    P                                      p,
    typename P::padded_multi_index_t const midx,
    typename P::index_t const              i
) {
    typename P::value_type;
    typename P::size_type;
    typename P::index_t;
    typename P::container_t;
    typename P::data_layout_t;
    typename P::padded_multi_index_t;
    { P::halo_width() } -> std::same_as<typename P::size_type>;
    { p.data() } -> std::same_as<typename P::container_t&>;
    { p.data() } -> std::same_as<typename P::container_t&>;
    { cp.data() } -> std::same_as<typename P::container_t const&>;
    { cp[midx] } -> std::same_as<typename P::value_type const&>;
    { cp[i] } -> std::same_as<typename P::value_type const&>;
};

template <typename L>
concept PatchLayout = requires() {
    typename L::rank_t;
    typename L::index_t;
    typename L::size_type;
    typename L::data_layout_t;
    typename L::padded_layout_t;
    { L::rank() } -> std::same_as<typename L::rank_t>;
    { L::flat_size() } -> std::same_as<typename L::size_type>;
    { L::halo_width() } -> std::same_as<typename L::size_type>;
};

template <typename D>
concept Direction = requires(D const cd) {
    typename D::index_t;
    typename D::size_type;
    { D::rank() } -> std::same_as<typename D::size_type>;
    { D::elements() } -> std::same_as<typename D::size_type>;
    { D::unit_vector(cd) } -> std::same_as<typename D::vector_t>;
    { cd.dimension() } -> std::same_as<typename D::index_t>;
    // TODO: This is horrible
    { D::is_negative(cd) } -> std::same_as<bool>;
    { D::is_positive(cd) } -> std::same_as<bool>;
};

} // namespace amr::ndt::concepts

#endif // AMR_INCLUDED_NDT_CONCEPTS
