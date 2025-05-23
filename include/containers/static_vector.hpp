#ifndef AMR_INCLUDED_STATIC_VECTOR
#define AMR_INCLUDED_STATIC_VECTOR

#include "casts.hpp"
#include <array>
#include <concepts>
#include <ostream>
#include <type_traits>

namespace detail::containers
{

template <auto N, typename Value_Type>
    requires(std::integral<decltype(N)>)
struct static_vector
{
    using value_type                         = Value_Type;
    using size_type                          = decltype(N);
    using index_t                            = size_type;
    inline static constexpr auto s_dimension = N;
    using container_t                        = std::array<value_type, s_dimension>;
    using const_iterator                     = typename container_t::const_iterator;
    using iterator                           = typename container_t::iterator;

    constexpr auto assert_in_bounds([[maybe_unused]] index_t const idx) const noexcept
        -> void
    {
        assert(idx < s_dimension);
        if constexpr (std::is_signed_v<index_t>)
        {
            assert(idx >= index_t{});
        }
    }

    [[nodiscard]]
    constexpr auto value() noexcept -> container_t&
    {
        return value_;
    }

    [[nodiscard]]
    constexpr auto value() const noexcept -> const container_t&
    {
        return value_;
    }

    [[nodiscard]]
    constexpr auto operator[](std::size_t idx) const noexcept -> value_type
    {
        assert_in_bounds(idx);
        return value_[idx];
    }

    [[nodiscard]]
    constexpr auto operator[](std::size_t idx) noexcept -> value_type&
    {
        assert_in_bounds(idx);
        return value_[idx];
    }

    [[nodiscard]]
    constexpr auto cbegin() const noexcept -> container_t::const_iterator
    {
        return std::cbegin(value_);
    }

    [[nodiscard]]
    constexpr auto cend() const noexcept -> container_t::const_iterator
    {
        return std::cend(value_);
    }

    [[nodiscard]]
    constexpr auto begin() const noexcept -> container_t::const_iterator
    {
        return std::begin(value_);
    }

    [[nodiscard]]
    constexpr auto end() const noexcept -> container_t::const_iterator
    {
        return std::end(value_);
    }

    [[nodiscard]]
    constexpr auto begin() noexcept -> container_t::iterator
    {
        return std::begin(value_);
    }

    [[nodiscard]]
    constexpr auto end() noexcept -> container_t::iterator
    {
        return std::end(value_);
    }

    constexpr auto operator<=>(static_vector const&) const noexcept = default;

public:
    container_t value_;
};

template <auto N, typename Value_Type>
    requires(std::integral<decltype(N)>)
auto operator<<(std::ostream& os, static_vector<N, Value_Type> const& pv) noexcept
    -> std::ostream&
{
    os << "{ ";
    std::size_t n{ 0 };
    for (auto const v : pv)
    {
        os << v << (++n != N ? ", " : " ");
    }
    os << '}';

    return os;
}

} // namespace detail::containers

#endif // AMR_INCLUDED_STATIC_VECTOR
