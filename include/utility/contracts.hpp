#ifndef INCLUDED_UTILITY_CONTRACTS
#define INCLUDED_UTILITY_CONTRACTS

#include <concepts>
#include <cstdlib>

#ifndef NDEBUG
#    define CONTRACTS_CHECK(expr) ::utility::contracts::check(static_cast<bool>(expr))
#    define CONTRACTS_CHECK_RANGE(i, low, high) \
        ::utility::contracts::check_range((i), (low), (high))
#    define CONTRACTS_CHECK_INDEX(i, size) ::utility::contracts::check_index((i), (size))
#else
#    define CONTRACTS_CHECK(expr)               ((void)0)
#    define CONTRACTS_CHECK_RANGE(i, low, high) ((void)0)
#    define CONTRACTS_CHECK_INDEX(i, size)      ((void)0)
#endif

namespace utility::contracts
{

[[noreturn]]
constexpr auto contract_fail() -> void
{
    if (std::is_constant_evaluated())
    {
        throw "contract violation";
    }
    else
    {
        std::abort();
    }
}

constexpr auto check(bool cond) noexcept -> void
{
    if (!cond) contract_fail();
}

template <std::integral I>
constexpr auto check_range(I const& i, I const& low, I const& high) noexcept -> void
{
    check((i >= low) && (i < high));
}

template <std::integral I>
constexpr auto check_index(I const& i, I const& size) noexcept -> void
{
    if constexpr (std::is_signed_v<I>)
    {
        check((i >= I{}) && (i < size));
    }
    else
    {
        check(i < size);
    }
}

} // namespace utility::contracts

#endif // INCLUDED_UTILITY_CONTRACTS
