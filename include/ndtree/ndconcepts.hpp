#ifndef AMR_INCLUDED_NDCONCEPTS
#define AMR_INCLUDED_NDCONCEPTS

#include <concepts>
#include <functional>
#include <optional>
#include <string_view>

namespace amr::ndt::concepts
{

template <typename T>
concept MapType = requires {
    typename T::type;
    { T::index() } -> std::same_as<std::size_t>;
    { T::name() } -> std::same_as<std::string_view>;
};

namespace detail
{

template <typename>
constexpr bool map_type_tuple_impl = false;
template <template <class...> class Tuple, class... Types>
    requires(MapType<Types> && ...)
constexpr bool map_type_tuple_impl<Tuple<Types...>> = true;

} // namespace detail

template <typename T>
concept MapTypeTuple = detail::map_type_tuple_impl<T>;

template <typename T>
concept DeconstructibleType = requires { typename T::deconstructed_types_map_t; } &&
                              MapTypeTuple<typename T::deconstructed_types_map_t>;

template <typename I>
concept PatchIndex =
    requires(const I i) {
        { I::rank() } -> std::same_as<typename I::size_type>;
        { I::fanout() } -> std::same_as<typename I::size_type>;
        { I::nd_fanout() } -> std::same_as<typename I::size_type>;
        { I::max_depth() } -> std::integral;
        { I::root() } -> std::same_as<I>;
        { I::parent_of(i) } -> std::same_as<I>;
        { I::child_of(i, std::declval<typename I::offset_t>()) } -> std::same_as<I>;
        {
            I::neighbor_at(i, std::declval<typename I::direction_t>())
        } -> std::same_as<std::optional<I>>;
        { I::offset_of(i) } -> std::same_as<typename I::offset_t>;
        {
            I::offset(i, std::declval<typename I::offset_t>())
        } -> std::same_as<I>; // TODO: Rethink
        { I::level(i) } -> std::same_as<typename I::level_t>;
        { i.id() } -> std::same_as<typename I::mask_t>;
        { i.repr() } -> std::convertible_to<std::string_view>;
        { std::less{}(i, i) } -> std::convertible_to<bool>;
    } && std::integral<typename I::size_type> &&
    std::unsigned_integral<typename I::mask_t> && std::equality_comparable<I>;

template <typename P>
concept Patch = requires(P const cp, P p) {
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
    {
        cp[std::declval<typename P::padded_multi_index_t>()]
    } -> std::same_as<typename P::value_type const&>;
    {
        cp[std::declval<typename P::index_t>()]
    } -> std::same_as<typename P::value_type const&>;
};

template <typename L>
concept PatchLayout = requires {
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
    { cd.is_negative() } -> std::same_as<bool>;
    { cd.is_positive() } -> std::same_as<bool>;
};

// TODO: Extend
template <typename T>
concept TreeType =
    requires(T t) {
        typename T::linear_index_t;
        typename T::patch_layout_t;
        typename T::deconstructed_raw_map_types_t;
    } && PatchLayout<typename T::patch_layout_t> &&
    MapTypeTuple<typename T::deconstructed_raw_map_types_t>;

template<typename HEO>
concept HaloExchangeOperator = requires{
    HEO::boundary;
    HEO::same;
    HEO::finer;
    HEO::coarser;
};

} // namespace amr::ndt::concepts

#endif // AMR_INCLUDED_NDT_CONCEPTS
