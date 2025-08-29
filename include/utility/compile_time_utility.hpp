#pragma once

#include <array>
#include <type_traits>

namespace utility::compile_time_utility
{

template <typename T, std::size_t N, typename Fn, typename... Args>
    requires std::is_invocable_r_v<T, Fn, std::size_t, Args...>
[[nodiscard]]
constexpr auto array_factory(Fn&& fn, Args&&... args) noexcept
    -> std::array<std::invoke_result_t<Fn, std::size_t, Args...>, N>
{
    return
        []<std::
               size_t... Indices>(Fn&& f, Args&&... a, std::index_sequence<Indices...>) {
            return std::array<T, N>{f(Indices, std::forward<Args>(a)...)...};
        }(std::forward<Fn>(fn),
          std::forward<Args>(args)...,
          std::make_index_sequence<N>{});
}

template <typename T, std::size_t N, typename Fn, typename... Args>
    requires std::is_invocable_r_v<T, Fn, Args...>
[[nodiscard]]
constexpr auto array_factory(Fn&& fn, Args&&... args) noexcept
    -> std::array<T, N>
{
    return
        []<std::
               size_t... Indices>(Fn&& f, Args&&... a, std::index_sequence<Indices...>) {
            return std::array<T, N>{
                {(static_cast<void>(Indices), f(std::forward<Args>(a)...))...}
            };
        }(std::forward<Fn>(fn),
          std::forward<Args>(args)...,
          std::make_index_sequence<N>{});
}

template <typename T, std::size_t N>
[[nodiscard]]
constexpr auto array_factory(T const& value) noexcept -> std::array<T, N>
{
    return
        []<std::
               size_t... Indices>(T const& v, std::index_sequence<Indices...>) {
            return std::array<T, N>{((void)Indices, v)...};
        }(value, std::make_index_sequence<N>{});
}

} // namespace utility::compile_time_utility
