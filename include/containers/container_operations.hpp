#ifndef AMR_INCLUDED_CONTAINER_OPERATIONS
#define AMR_INCLUDED_CONTAINER_OPERATIONS

#include "error_handling.hpp"
#include <concepts>
#include <functional>
#include <ranges>
#include <type_traits>

namespace containers::vector_operations
{

namespace detail
{

template <typename V>
concept Vector = requires(V v, typename V::size_type i) {
    typename V::value_type;
    v[i];
} && std::ranges::sized_range<V> && std::is_trivially_constructible_v<V>;

template <typename T1, typename T2>
    requires Vector<T1> || Vector<T2>
struct common_type;

template <Vector V>
struct common_type<V, V>
{
    using type = V;
};

template <Vector V, typename T>
    requires std::is_arithmetic_v<T>
struct common_type<V, T>
{
    using type = V;
};

template <typename T, Vector V>
    requires std::is_arithmetic_v<T>
struct common_type<T, V>
{
    using type = V;
};

template <typename T1, typename T2>
using common_type_t = typename common_type<T1, T2>::type;

} // namespace detail

constexpr auto operator+(auto&& lhs, auto&& rhs) noexcept -> auto
    requires detail::Vector<std::remove_cvref_t<decltype(lhs)>> ||
             detail::Vector<std::remove_cvref_t<decltype(rhs)>>
{
    return operator_impl(
        std::forward<decltype(lhs)>(lhs), std::forward<decltype(rhs)>(rhs), std::plus{}
    );
}

constexpr auto operator-(auto&& lhs, auto&& rhs) noexcept -> auto
    requires detail::Vector<std::remove_cvref_t<decltype(lhs)>> ||
             detail::Vector<std::remove_cvref_t<decltype(rhs)>>
{
    return operator_impl(
        std::forward<decltype(lhs)>(lhs), std::forward<decltype(rhs)>(rhs), std::minus{}
    );
}

constexpr auto operator*(auto&& lhs, auto&& rhs) noexcept -> auto
    requires detail::Vector<std::remove_cvref_t<decltype(lhs)>> ||
             detail::Vector<std::remove_cvref_t<decltype(rhs)>>
{
    return operator_impl(
        std::forward<decltype(lhs)>(lhs),
        std::forward<decltype(rhs)>(rhs),
        std::multiplies{}
    );
}

constexpr auto operator/(auto&& lhs, auto&& rhs) noexcept -> auto
    requires detail::Vector<std::remove_cvref_t<decltype(lhs)>> ||
             detail::Vector<std::remove_cvref_t<decltype(rhs)>>
{
    return operator_impl(
        std::forward<decltype(lhs)>(lhs), std::forward<decltype(rhs)>(rhs), std::divides{}
    );
}

constexpr auto max(auto&& lhs, auto&& rhs) noexcept -> auto
    requires detail::Vector<std::remove_cvref_t<decltype(lhs)>> ||
             detail::Vector<std::remove_cvref_t<decltype(rhs)>>
{
    return operator_impl(
        std::forward<decltype(lhs)>(lhs),
        std::forward<decltype(rhs)>(rhs),
        [](auto&& l, auto&& r) constexpr noexcept -> decltype(auto)
        {
            return std::less{}(std::forward<decltype(l)>(l), std::forward<decltype(r)>(r))
                       ? std::forward<decltype(r)>(r)
                       : std::forward<decltype(l)>(l);
        }
    );
}

constexpr auto min(auto&& lhs, auto&& rhs) noexcept -> auto
    requires detail::Vector<std::remove_cvref_t<decltype(lhs)>> ||
             detail::Vector<std::remove_cvref_t<decltype(rhs)>>
{
    return operator_impl(
        std::forward<decltype(lhs)>(lhs),
        std::forward<decltype(rhs)>(rhs),
        [](auto&& l, auto&& r) constexpr noexcept -> decltype(auto)
        {
            return std::less{}(std::forward<decltype(l)>(l), std::forward<decltype(r)>(r))
                       ? std::forward<decltype(l)>(l)
                       : std::forward<decltype(r)>(r);
        }
    );
}

template <typename T1, typename T2>
[[nodiscard]]
constexpr auto operator_impl(T1&& lhs, T2&& rhs, auto&& binary_op) noexcept
    -> detail::common_type_t<std::remove_cvref_t<T1>, std::remove_cvref_t<T2>>
    requires detail::Vector<std::remove_cvref_t<decltype(lhs)>> ||
             detail::Vector<std::remove_cvref_t<decltype(rhs)>>
{
    using a_type      = std::remove_cvref_t<T1>;
    using b_type      = std::remove_cvref_t<T2>;
    using common_type = detail::common_type_t<a_type, b_type>;

    static_assert(std::is_trivially_constructible_v<common_type>);
    static_assert(std::ranges::sized_range<common_type>);

    constexpr auto at_idx =
        [](auto&& v, std::integral auto idx) constexpr noexcept -> decltype(auto)
        requires(
            detail::Vector<std::remove_cvref_t<decltype(v)>> ||
            std::is_arithmetic_v<std::remove_cvref_t<decltype(v)>>
        )
    {
        using v_type = std::remove_cvref_t<decltype(v)>;
        if constexpr (std::is_arithmetic_v<v_type>)
        {
            return v;
        }
        else if constexpr (detail::Vector<v_type>)
        {
            return v[idx];
        }
        else if constexpr (std::ranges::range<v_type>)
        {
            return v[idx];
        }
        else
        {
            utility::error_handling::assert_unreachable();
        }
    };

    common_type ret{};
    for (auto i = typename common_type::size_type{}; i != ret.size(); ++i)
    {
        ret[i] = binary_op(
            at_idx(std::forward<decltype(lhs)>(lhs), i),
            at_idx(std::forward<decltype(rhs)>(rhs), i)
        );
    }
    return ret;
}

} // namespace containers::vector_operations

#endif // AMR_INCLUDED_CONTAINER_OPERATIONS
