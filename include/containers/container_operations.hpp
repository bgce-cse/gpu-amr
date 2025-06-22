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

template <concepts::Vector V, typename T>
    requires std::is_arithmetic_v<T>
struct common_type<V, T>
{
    static_assert(std::is_same_v<
                  typename V::value_type,
                  std::common_type_t<typename V::value_type, T>>);
    using type = V;
};

template <typename T, concepts::Vector V>
    requires std::is_arithmetic_v<T>
struct common_type<T, V>
{
    static_assert(std::is_same_v<
                  typename V::value_type,
                  std::common_type_t<typename V::value_type, T>>);
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

constexpr auto operator+(auto const& lhs, auto const& rhs) noexcept -> auto
    requires concepts::Vector<std::remove_cvref_t<decltype(lhs)>> ||
             concepts::Vector<std::remove_cvref_t<decltype(rhs)>>
{
    return operator_impl(lhs, rhs, std::plus{});
}

constexpr auto operator-(auto const& lhs, auto const& rhs) noexcept -> auto
    requires concepts::Vector<std::remove_cvref_t<decltype(lhs)>> ||
             concepts::Vector<std::remove_cvref_t<decltype(rhs)>>
{
    return operator_impl(lhs, rhs, std::minus{});
}

constexpr auto operator*(auto const& lhs, auto const& rhs) noexcept -> auto
    requires concepts::Vector<std::remove_cvref_t<decltype(lhs)>> ||
             concepts::Vector<std::remove_cvref_t<decltype(rhs)>>
{
    return operator_impl(lhs, rhs, std::multiplies{});
}

constexpr auto operator/(auto const& lhs, auto const& rhs) noexcept -> auto
    requires concepts::Vector<std::remove_cvref_t<decltype(lhs)>> ||
             concepts::Vector<std::remove_cvref_t<decltype(rhs)>>
{
    return operator_impl(lhs, rhs, std::divides{});
}

constexpr auto max(auto const& lhs, auto const& rhs) noexcept -> auto
    requires concepts::Vector<std::remove_cvref_t<decltype(lhs)>> ||
             concepts::Vector<std::remove_cvref_t<decltype(rhs)>>
{
    return operator_impl(
        lhs,
        rhs,
        [](auto const& l, auto const& r) constexpr noexcept -> decltype(auto)
        { return std::less{}(l, r) ? r : l; }
    );
}

constexpr auto min(auto const& lhs, auto const& rhs) noexcept -> auto
    requires concepts::Vector<std::remove_cvref_t<decltype(lhs)>> ||
             concepts::Vector<std::remove_cvref_t<decltype(rhs)>>
{
    return operator_impl(
        lhs,
        rhs,
        [](auto const& l, auto const& r) constexpr noexcept -> decltype(auto)
        { return std::less{}(l, r) ? l : r; }
    );
}

[[nodiscard]]
constexpr auto operator_impl(auto const& lhs, auto const& rhs, auto&& binary_op) noexcept
    -> detail::common_type_t<
        std::remove_cvref_t<decltype(lhs)>,
        std::remove_cvref_t<decltype(rhs)>>
    requires concepts::Vector<std::remove_cvref_t<decltype(lhs)>> ||
             concepts::Vector<std::remove_cvref_t<decltype(rhs)>>
{
    using a_type      = std::remove_cvref_t<decltype(lhs)>;
    using b_type      = std::remove_cvref_t<decltype(rhs)>;
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
    for (auto i = typename common_type::size_type{}; i != ret.size(); ++i)
    {
        ret[i] = std::invoke(
            std::forward<decltype(binary_op)>(binary_op), at_idx(lhs, i), at_idx(rhs, i)
        );
    }
    return ret;
}

} // namespace amr::containers

#endif // AMR_INCLUDED_CONTAINER_OPERATIONS
