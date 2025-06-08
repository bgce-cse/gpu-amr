#ifndef AMR_INCLUDED_NDCONCEPTS
#define AMR_INCLUDED_NDCONCEPTS

#include <iterator>
#include <optional>
#include <type_traits>

namespace amr::tree::concepts
{

template <typename T>
concept Point = requires(T t) {
    typename T::value_type;
    T::s_dimension;
    t[std::declval<int>()];
    std::begin(t);
    std::end(t);
};

template <typename T>
concept Boundary = requires(T t) {
    t.min();
    t.max();
    t.mid(0);
};

template <typename T>
concept Cell = requires(T t) {
    typename T::value_type;
    T::s_dimension;
    { t.position() } -> std::convertible_to<typename T::position_t>;
    t.properties();
    { merge(std::array{ t, t }) } -> std::same_as<std::optional<T>>;
} && std::is_destructible_v<T>;

} // namespace amr::tree::concepts

#endif // AMR_INCLUDED_NDT_CONCEPTS
