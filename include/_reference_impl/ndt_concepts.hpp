#ifndef GPU_ARM_NDT_CONCEPTS
#define GPU_ARM_NDT_CONCEPTS

#include <iterator>
#include <optional>
#include <type_traits>

namespace ndt_concepts
{

template <typename T>
concept point = requires(T t) {
    typename T::value_type;
    T::s_dimension;
    t[0];
    std::begin(t);
    std::end(t);
};

template <typename T>
concept boundary = requires(T t) {
    t.min();
    t.max();
    t.mid(0);
};

template <typename T>
concept cell = requires(T t) {
    typename T::value_type;
    T::s_dimension;
    { t.position() } -> std::convertible_to<typename T::position_t>;
    t.properties();
    { merge(std::array{ t, t }) } -> std::same_as<std::optional<T>>;
} && std::is_destructible_v<T>;

} // namespace ndt_concepts

#endif // GPU_ARM_NDT_CONCEPTS
