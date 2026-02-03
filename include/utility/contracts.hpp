#ifndef INCLUDED_UTILITY_CONTRACTS
#define INCLUDED_UTILITY_CONTRACTS

#include "utility_concepts.hpp"
#include <concepts>
#include <utility>

#ifndef NDEBUG
#    define CONTRACTS_CHECK(expr) utility::contracts::check(expr)
#else
#    define CONTRACTS_CHECK(expr) ((void)0)
#endif

namespace utility::contracts
{

template <concepts::Arithmetic T>
[[gnu::always_inline]]
constexpr auto check_range(T const& i, T const& low, T const& high) noexcept -> void
{
    if (std::is_constant_evaluated())
    {
        if constexpr (std::is_signed_v<T>)
        {
            if (i < low) std::unreachable();
        }
        if (i >= high) std::unreachable();
    }
#ifndef NDEBUG
    else
    {
        if constexpr (std::is_signed_v<T>)
        {
            if (i < low) [[unlikely]]
                std::terminate();
        }
        if (i >= high) [[unlikely]]
            std::terminate();
    }
#else
    else
    {
        if constexpr (std::is_signed_v<T>)
        {
            if (i < low) std::unreachable();
        }
        if (i >= high) std::unreachable();
    }
#endif
}

template <concepts::Arithmetic T>
[[gnu::always_inline]]
constexpr auto check_index(T const& i, T const& size) noexcept -> void
{
    check_range(i, T{}, size);
}

[[gnu::always_inline]]
constexpr auto check(auto&& cond) noexcept -> void
    requires(std::is_convertible_v<std::remove_cvref_t<decltype(cond)>, bool>)
{
    const auto b = static_cast<bool>(cond);
    if (std::is_constant_evaluated())
    {
        if (!b) std::unreachable();
    }
#ifndef NDEBUG
    else
    {
        if (!b) [[unlikely]]
            std::terminate();
    }
#else
    else
    {
        if (!b) std::unreachable();
    }
#endif
}

} // namespace utility::contracts

#endif // INCLUDED_UTILITY_CONTRACTS
