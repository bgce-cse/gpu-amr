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
        { I::dimension() } -> std::same_as<typename I::size_type>;
        { I::fanout() } -> std::same_as<typename I::size_type>;
        { I::nd_fanout() } -> std::same_as<typename I::size_type>;
        { I::max_depth() } -> std::integral;
        { I::root() } -> std::same_as<I>;
        { I::parent_of(i) } -> std::same_as<I>;
        { I::child_of(i, offset) } -> std::same_as<I>;
        { I::neighbour_at(i, d) } -> std::same_as<std::optional<I>>;
        { I::offset_of(i) } -> std::same_as<typename I::offset_t>;
        { I::offset(i, offset) } -> std::same_as<I>; // TODO: Rethink
        { I::level(i) } -> std::same_as<typename I::level_t>;
        { i.id() } -> std::same_as<typename I::mask_t>;
        { i.repr() } -> std::same_as<std::string>;
        { std::less{}(i, i) } -> std::convertible_to<bool>;
    } &&
    std::integral<typename I::size_type> && std::unsigned_integral<typename I::mask_t> &&
    std::equality_comparable<I>;

template <typename T>
concept StaticLayout = requires {
    // Static constexpr members
    { T::s_rank } -> std::convertible_to<std::size_t>;
    { T::s_flat_size } -> std::convertible_to<std::size_t>;
    
    // Linear index functionality - now matches static_tensor's interface
    requires requires(typename T::index_t const (&idxs)[T::s_rank]) {
        { T::linear_index(idxs) } -> std::convertible_to<typename T::index_t>;
    };
    
    requires requires(typename T::multi_index_t const& multi_idx) {
        { T::linear_index(multi_idx) } -> std::convertible_to<typename T::index_t>;
    };
    
    // Required type aliases
    typename T::size_type;
    typename T::index_t;
    typename T::multi_index_t;
};

} // namespace amr::ndt::concepts

#endif // AMR_INCLUDED_NDT_CONCEPTS
