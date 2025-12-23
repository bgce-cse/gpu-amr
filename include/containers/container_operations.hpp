#ifndef AMR_INCLUDED_CONTAINER_OPERATIONS
#define AMR_INCLUDED_CONTAINER_OPERATIONS

#include "container_concepts.hpp"
#include "utility/error_handling.hpp"
#include "utility/utility_concepts.hpp"
#include <concepts>
#include <functional>
#include <ranges>
#include <type_traits>

namespace amr::containers
{

namespace detail
{

template <typename T1, typename T2>
    requires concepts::StaticContainer<T1> || concepts::StaticContainer<T2>
struct common_type;

template <concepts::Vector V>
struct common_type<V, V>
{
    using type = V;
};

template <concepts::StaticContainer C1, concepts::StaticContainer C2>
    requires(C1::rank() == C2::rank() && C1::sizes() == C2::sizes())
struct common_type<C1, C2>
{
    using type = C1::template rebind_t<
        std::common_type_t<typename C1::value_type, typename C2::value_type>>;
};

template <concepts::StaticContainer C, utility::concepts::Arithmetic T>
    requires(std::is_convertible_v<typename C::value_type, T>)
struct common_type<C, T>
{
    using type = C::template rebind_t<std::common_type_t<typename C::value_type, T>>;
};

template <utility::concepts::Arithmetic T, concepts::StaticContainer C>
    requires(std::is_convertible_v<typename C::value_type, T>)
struct common_type<T, C>
{
    using type = C::template rebind_t<std::common_type_t<typename C::value_type, T>>;
};

template <typename T1, typename T2>
using common_type_t = typename common_type<T1, T2>::type;

} // namespace detail

constexpr auto operator+(auto const& lhs, auto const& rhs) noexcept -> auto
    requires concepts::StaticContainer<std::remove_cvref_t<decltype(lhs)>> ||
             concepts::StaticContainer<std::remove_cvref_t<decltype(rhs)>>
{
    return operator_impl(lhs, rhs, std::plus{});
}

constexpr auto operator-(auto const& lhs, auto const& rhs) noexcept -> auto
    requires concepts::StaticContainer<std::remove_cvref_t<decltype(lhs)>> ||
             concepts::StaticContainer<std::remove_cvref_t<decltype(rhs)>>
{
    return operator_impl(lhs, rhs, std::minus{});
}

constexpr auto operator*(auto const& lhs, auto const& rhs) noexcept -> auto
    requires concepts::StaticContainer<std::remove_cvref_t<decltype(lhs)>> ||
             concepts::StaticContainer<std::remove_cvref_t<decltype(rhs)>>
{
    return operator_impl(lhs, rhs, std::multiplies{});
}

constexpr auto operator/(auto const& lhs, auto const& rhs) noexcept -> auto
    requires concepts::StaticContainer<std::remove_cvref_t<decltype(lhs)>> ||
             concepts::StaticContainer<std::remove_cvref_t<decltype(rhs)>>
{
    return operator_impl(lhs, rhs, std::divides{});
}

constexpr auto max(auto const& lhs, auto const& rhs) noexcept -> auto
    requires concepts::StaticContainer<std::remove_cvref_t<decltype(lhs)>> ||
             concepts::StaticContainer<std::remove_cvref_t<decltype(rhs)>>
{
    return operator_impl(
        lhs,
        rhs,
        [](auto const& l, auto const& r) constexpr noexcept -> decltype(auto)
        { return std::less{}(l, r) ? r : l; }
    );
}

constexpr auto min(auto const& lhs, auto const& rhs) noexcept -> auto
    requires concepts::StaticContainer<std::remove_cvref_t<decltype(lhs)>> ||
             concepts::StaticContainer<std::remove_cvref_t<decltype(rhs)>>
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
    requires concepts::StaticContainer<std::remove_cvref_t<decltype(lhs)>> ||
             concepts::StaticContainer<std::remove_cvref_t<decltype(rhs)>>
{
    using a_type      = std::remove_cvref_t<decltype(lhs)>;
    using b_type      = std::remove_cvref_t<decltype(rhs)>;
    using common_type = detail::common_type_t<a_type, b_type>;

    static_assert(std::is_trivially_constructible_v<common_type>);
    static_assert(std::ranges::sized_range<common_type>);

    constexpr auto at_idx = [](
                                auto const& v, std::integral auto const idx
                            ) constexpr noexcept -> decltype(auto)
        requires(
            concepts::StaticContainer<std::remove_cvref_t<decltype(v)>> ||
            std::is_arithmetic_v<std::remove_cvref_t<decltype(v)>>
        )
    {
        using v_type = std::remove_cvref_t<decltype(v)>;
        if constexpr (std::is_arithmetic_v<v_type>)
        {
            return v;
        }
        else if constexpr (concepts::StaticContainer<v_type>)
        {
            return v.underlying_at(idx);
        }
        else
        {
            utility::error_handling::assert_unreachable();
        }
    };

    common_type ret{};
    for (auto i = typename common_type::size_type{}; i != ret.elements(); ++i)
    {
        ret.underlying_at(i) = std::invoke(
            std::forward<decltype(binary_op)>(binary_op), at_idx(lhs, i), at_idx(rhs, i)
        );
    }
    return ret;
}

} // namespace amr::containers

#endif // AMR_INCLUDED_CONTAINER_OPERATIONS
