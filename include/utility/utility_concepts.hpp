#pragma once

#include <chrono>
#include <concepts>
#include <type_traits>

namespace utility::concepts
{

template <typename T>
concept Initializable = requires { T::init(); };

template <typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

template <typename T>
concept RandomDistribution = requires(T t) {
    typename T::param_type;
    typename T::result_type;
    { t() } -> std::same_as<typename T::result_type>;
    requires std::constructible_from<T, typename T::param_type>;
};

template <typename T>
concept Duration = requires(T t) { std::chrono::duration_cast<std::chrono::seconds>(t); };

template <typename T, typename... Ts>
concept is_any = std::disjunction_v<std::is_same<T, Ts>...>;

template <typename T, typename... Ts>
concept are_same = std::conjunction_v<std::is_same<T, Ts>...>;

} // namespace utility::concepts
