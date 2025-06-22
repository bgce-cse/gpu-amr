#ifndef AMR_INCLUDED_NDCONCEPTS
#define AMR_INCLUDED_NDCONCEPTS

#include <concepts>
#include <functional>
#include <iterator>
#include <optional>
#include <type_traits>

namespace amr::ndt::concepts
{

template <typename I>
concept NodeIndex =
    requires(
        const I                       i,
        const typename I::direction_t d,
        const typename I::offset_t    offset
    ) {
        { I::dimension() } -> std::same_as<typename I::size_type>;
        { I::fanout() } -> std::same_as<typename I::size_type>;
        { I::nd_fanout() } -> std::same_as<typename I::size_type>;
        { I::max_depth() } -> std::integral;
        { I::zeroth_generation() } -> std::same_as<I>;
        { I::parent_of(i) } -> std::same_as<I>;
        { I::child_of(i, offset) } -> std::same_as<I>;
        { I::neighbour_at(i, d) } -> std::same_as<I>;
        { I::offset_of(i) } -> std::same_as<typename I::offset_t>;
        { I::offset(i, offset) } -> std::same_as<I>; // TODO: Rethink
        { std::less{}(i, i) } -> std::convertible_to<bool>;
    } &&
    std::integral<typename I::size_type> && std::equality_comparable<I>;

} // namespace amr::ndt::concepts

#endif // AMR_INCLUDED_NDT_CONCEPTS
