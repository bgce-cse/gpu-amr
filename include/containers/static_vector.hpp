#ifndef AMR_INCLUDED_STATIC_VECTOR
#define AMR_INCLUDED_STATIC_VECTOR

#include "casts.hpp"
#include <array>
#include <concepts>
#include <iterator>
#include <ostream>
#include <type_traits>

namespace amr::containers
{

template <auto N, typename Value_Type>
    requires(std::integral<decltype(N)>)
struct static_vector
{
    using value_type     = Value_Type;
    using size_type      = decltype(N);
    using const_iterator = value_type const*;
    using iterator       = value_type*;

    // TODO: Alignment
    static constexpr auto s_size = N;

    static_assert(std::is_trivially_copyable_v<static_vector>);
    static_assert(std::is_standard_layout_v<static_vector>);

    [[nodiscard]]
    constexpr static auto size() noexcept -> size_type
    {
        return s_size;
    }

    [[nodiscard]]
    constexpr auto operator[](size_type const idx) const noexcept -> value_type const&
    {
        assert_in_bounds(idx);
        return value_[idx];
    }

    [[nodiscard]]
    constexpr auto operator[](size_type const idx) noexcept -> value_type&
    {
        assert_in_bounds(idx);
        return value_[idx];
    }

    [[nodiscard]]
    constexpr auto cbegin() const noexcept -> const_iterator
    {
        return std::cbegin(value_);
    }

    [[nodiscard]]
    constexpr auto cend() const noexcept -> const_iterator
    {
        return std::cend(value_);
    }

    [[nodiscard]]
    constexpr auto begin() const noexcept -> const_iterator
    {
        return std::begin(value_);
    }

    [[nodiscard]]
    constexpr auto end() const noexcept -> const_iterator
    {
        return std::end(value_);
    }

    [[nodiscard]]
    constexpr auto begin() noexcept -> iterator
    {
        return std::begin(value_);
    }

    [[nodiscard]]
    constexpr auto end() noexcept -> iterator
    {
        return std::end(value_);
    }

    constexpr auto assert_in_bounds([[maybe_unused]] size_type const idx) const noexcept
        -> void
    {
        assert(idx < s_size);
        if constexpr (std::is_signed_v<size_type>)
        {
            assert(idx >= size_type{});
        }
    }

    constexpr auto operator<=>(static_vector const&) const noexcept = default;

public:
    value_type value_[s_size];
};

template <auto N, typename Value_Type>
    requires(std::integral<decltype(N)>)
auto operator<<(std::ostream& os, static_vector<N, Value_Type> const& v) noexcept
    -> std::ostream&
{
    os << "{ ";
    std::size_t n{ 0 };
    for (auto const& e : v)
    {
        os << e << (++n != N ? ", " : " ");
    }
    os << '}';
    return os;
}

} // namespace amr::containers

#endif // AMR_INCLUDED_STATIC_VECTOR
