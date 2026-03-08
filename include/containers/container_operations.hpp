#ifndef AMR_INCLUDED_CONTAINER_OPERATIONS
#define AMR_INCLUDED_CONTAINER_OPERATIONS

#include "container_concepts.hpp"
#include "utility/error_handling.hpp"
#include <concepts>
#include <functional>
#include <ranges>
#include <type_traits>

namespace amr::containers
{

namespace detail
{

// Helper to check if scalar can be converted to vector's value_type
template <typename T, typename Scalar>
concept ScalarCompatibleWith =
    std::is_arithmetic_v<Scalar> && std::convertible_to<Scalar, T>;

template <typename T1, typename T2>
    requires concepts::Vector<T1> || concepts::Vector<T2>
struct common_type;

template <concepts::Vector V>
struct common_type<V, V>
{
    using type = V;
};

template <concepts::Vector V1, concepts::Vector V2>
    requires(std::is_same_v<
             typename V1::value_type,
             std::common_type_t<typename V1::value_type, typename V2::value_type>>)
struct common_type<V1, V2>
{
    using type = V1;
};

template <concepts::Vector V1, concepts::Vector V2>
    requires(std::is_same_v<
             typename V2::value_type,
             std::common_type_t<typename V1::value_type, typename V2::value_type>>)
struct common_type<V1, V2>
{
    using type = V2;
};

// Modified: Handle arithmetic types that are compatible with the vector's value_type
template <concepts::Vector V, typename T>
    requires std::is_arithmetic_v<T>
struct common_type<V, T>
{
    // Check if T is compatible with V's value_type (allowing nested vectors)
    // If V::value_type is itself a Vector, we need recursive compatibility
    static_assert(
        std::convertible_to<T, typename V::value_type> ||
            concepts::Vector<typename V::value_type>,
        "Scalar type must be convertible to vector's value type"
    );
    using type = V;
};

template <typename T, concepts::Vector V>
    requires std::is_arithmetic_v<T>
struct common_type<T, V>
{
    static_assert(
        std::convertible_to<T, typename V::value_type> ||
            concepts::Vector<typename V::value_type>,
        "Scalar type must be convertible to vector's value type"
    );
    using type = V;
};

template <concepts::Vector V, std::ranges::range R>
    requires(!concepts::Vector<R>)
struct common_type<V, R>
{
    static_assert(
        std::is_same_v<
            typename V::value_type,
            std::common_type_t<typename V::value_type, std::ranges::range_value_t<R>>>
    );
    using type = V;
};

template <std::ranges::range R, concepts::Vector V>
    requires(!concepts::Vector<R>)
struct common_type<R, V>
{
    static_assert(
        std::is_same_v<
            typename V::value_type,
            std::common_type_t<std::ranges::range_value_t<R>, typename V::value_type>>
    );
    using type = V;
};

template <typename T1, typename T2>
using common_type_t = typename common_type<T1, T2>::type;

} // namespace detail

template <typename L, typename R>
    requires concepts::Vector<L> || concepts::Vector<R>
constexpr auto operator+(L const& lhs, R const& rhs) noexcept -> auto
{
    return operator_impl(lhs, rhs, std::plus{});
}

template <typename L, typename R>
    requires concepts::Vector<L> || concepts::Vector<R>
constexpr auto operator-(L const& lhs, R const& rhs) noexcept -> auto
{
    return operator_impl(lhs, rhs, std::minus{});
}

template <typename L, typename R>
    requires concepts::Vector<L> || concepts::Vector<R>
constexpr auto operator*(L const& lhs, R const& rhs) noexcept -> auto
{
    return operator_impl(lhs, rhs, std::multiplies{});
}

template <typename L, typename R>
    requires concepts::Vector<L> || concepts::Vector<R>
constexpr auto operator/(L const& lhs, R const& rhs) noexcept -> auto
{
    return operator_impl(lhs, rhs, std::divides{});
}

template <typename L, typename R>
    requires concepts::Vector<L> || concepts::Vector<R>
constexpr auto max(L const& lhs, R const& rhs) noexcept -> auto
{
    return operator_impl(
        lhs,
        rhs,
        [](auto const& l, auto const& r) constexpr noexcept -> decltype(auto)
        { return std::less{}(l, r) ? r : l; }
    );
}

template <typename L, typename R>
    requires concepts::Vector<L> || concepts::Vector<R>
constexpr auto min(L const& lhs, R const& rhs) noexcept -> auto
{
    return operator_impl(
        lhs,
        rhs,
        [](auto const& l, auto const& r) constexpr noexcept -> decltype(auto)
        { return std::less{}(l, r) ? l : r; }
    );
}

template <typename L, typename R, typename BinaryOp>
    requires concepts::Vector<L> || concepts::Vector<R>
[[nodiscard]]
constexpr auto operator_impl(L const& lhs, R const& rhs, BinaryOp&& binary_op) noexcept
    -> detail::common_type_t<L, R>
{
    using a_type      = L;
    using b_type      = R;
    using common_type = detail::common_type_t<a_type, b_type>;

    static_assert(std::is_trivially_constructible_v<common_type>);
    static_assert(std::ranges::sized_range<common_type>);

    constexpr auto at_idx =
        [](auto const& v, std::integral auto idx) constexpr noexcept -> decltype(auto)
        requires(
            concepts::Vector<std::remove_cvref_t<decltype(v)>> ||
            std::ranges::range<std::remove_cvref_t<decltype(v)>> ||
            std::is_arithmetic_v<std::remove_cvref_t<decltype(v)>>
        )
    {
        using v_type = std::remove_cvref_t<decltype(v)>;
        if constexpr (std::is_arithmetic_v<v_type>)
        {
            return v;
        }
        else if constexpr (concepts::Vector<v_type>)
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
    for (auto i = typename common_type::size_type{}; i != ret.elements(); ++i)
    {
        ret[i] = std::invoke(
            std::forward<BinaryOp>(binary_op), at_idx(lhs, i), at_idx(rhs, i)
        );
    }
    return ret;
}

namespace detail
{

template <concepts::Vector L, typename R, typename BinaryOp>
constexpr auto compound_assign_impl(L& lhs, R const& rhs, BinaryOp&& op) noexcept -> L&
{
    constexpr auto at_idx =
        [](auto const& v, std::integral auto idx) constexpr noexcept -> decltype(auto)
        requires(
            concepts::Vector<std::remove_cvref_t<decltype(v)>> ||
            std::ranges::range<std::remove_cvref_t<decltype(v)>> ||
            std::is_arithmetic_v<std::remove_cvref_t<decltype(v)>>
        )
    {
        using v_type = std::remove_cvref_t<decltype(v)>;
        if constexpr (std::is_arithmetic_v<v_type>)
            return v;
        else
            return v[idx];
    };

    for (auto i = typename L::size_type{}; i != lhs.elements(); ++i)
    {
        lhs[i] = std::invoke(std::forward<BinaryOp>(op), lhs[i], at_idx(rhs, i));
    }
    return lhs;
}

} // namespace detail

template <concepts::Vector L, typename R>
constexpr auto operator+=(L& lhs, R const& rhs) noexcept -> L&
{
    return detail::compound_assign_impl(lhs, rhs, std::plus{});
}

template <concepts::Vector L, typename R>
constexpr auto operator-=(L& lhs, R const& rhs) noexcept -> L&
{
    return detail::compound_assign_impl(lhs, rhs, std::minus{});
}

template <concepts::Vector L, typename R>
constexpr auto operator*=(L& lhs, R const& rhs) noexcept -> L&
{
    return detail::compound_assign_impl(lhs, rhs, std::multiplies{});
}

template <concepts::Vector L, typename R>
constexpr auto operator/=(L& lhs, R const& rhs) noexcept -> L&
{
    return detail::compound_assign_impl(lhs, rhs, std::divides{});
}

} // namespace amr::containers

#endif // AMR_INCLUDED_CONTAINER_OPERATIONS