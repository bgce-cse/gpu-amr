#ifndef INCLUDED_UTILITY_CONTRACTS
#define INCLUDED_UTILITY_CONTRACTS

#include <cassert>
#include <concepts>
#include <cstdlib>
#include <iostream>
#include <utility>

#ifdef AMR_ENABLE_CONTRACTS
#    define CONTRACTS_CHECK(expr) ::utility::contracts::check((expr), __FILE__, __LINE__)
#    define CONTRACTS_CHECK_RANGE(i, low, high) \
        ::utility::contracts::check_range((i), (low), (high), __FILE__, __LINE__)
#    define CONTRACTS_CHECK_INDEX(i, size) \
        ::utility::contracts::check_index((i), (size), __FILE__, __LINE__)
#else
#    define CONTRACTS_CHECK(expr)               ((void)0)
#    define CONTRACTS_CHECK_RANGE(i, low, high) ((void)0)
#    define CONTRACTS_CHECK_INDEX(i, size)      ((void)0)
#endif

namespace utility::contracts
{

[[noreturn]]
constexpr auto contract_fail(const char* file, int line) -> void
{
    if (std::is_constant_evaluated())
    {
        #if defined(__CUDACC__) || __cplusplus < 202302L
            __builtin_unreachable();
        #else
            std::unreachable();
        #endif
    }
    else
    {
        std::cerr << "Contract violation at " << file << ":" << line << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

constexpr auto check(bool cond, const char* file, int line) noexcept -> void
{
    if (!cond) contract_fail(file, line);
}

template <std::integral I>
constexpr auto check_range(
    I const&    i,
    I const&    low,
    I const&    high,
    const char* file,
    int         line
) noexcept -> void
{
    check(false, file, line);
    // check((i >= low) && (i < high), file, line);
}

template <std::integral I>
constexpr auto check_index(I const& i, I const& size, const char* file, int line) noexcept
    -> void
{
    if constexpr (std::is_signed_v<I>)
    {
        check((i >= I{}) && (i < size), file, line);
    }
    else
    {
        check(i < size, file, line);
    }
}

} // namespace utility::contracts

#endif // INCLUDED_UTILITY_CONTRACTS
