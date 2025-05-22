#ifndef AMR_INCLUDED_STATIC_VECTOR
#define AMR_INCLUDED_STATIC_VECTOR

#include "casts.hpp"
#include <array>
#include <concepts>
#include <ostream>

namespace detail::containers
{

template <auto N, typename Value_Type>
    requires(std::integral<decltype(N)>)
struct physical_vector
{
    using value_type                         = Value_Type;
    using size_type                          = decltype(N);
    inline static constexpr auto s_dimension = N;
    using container_t                        = std::array<value_type, s_dimension>;
    using const_iterator                     = typename container_t::const_iterator;
    using iterator                           = typename container_t::iterator;

    inline auto assert_in_bounds([[maybe_unused]] std::integral auto const idx) const
        -> void
    {
        assert(idx < utility::casts::safe_cast<decltype(idx)>(s_dimension));
    }

    [[nodiscard]]
    auto value() noexcept -> container_t&
    {
        return value_;
    }

    [[nodiscard]]
    auto value() const noexcept -> const container_t&
    {
        return value_;
    }

    [[nodiscard]]
    auto operator[](std::size_t idx) const -> value_type
    {
        assert_in_bounds(idx);
        return value_[idx];
    }

    [[nodiscard]]
    auto operator[](std::size_t idx) -> value_type&
    {
        assert_in_bounds(idx);
        return value_[idx];
    }

    [[nodiscard]]
    auto cbegin() const -> container_t::const_iterator
    {
        return std::cbegin(value_);
    }

    [[nodiscard]]
    auto cend() const -> container_t::const_iterator
    {
        return std::cend(value_);
    }

    [[nodiscard]]
    auto begin() const noexcept -> container_t::const_iterator
    {
        return std::begin(value_);
    }

    [[nodiscard]]
    auto end() const noexcept -> container_t::const_iterator
    {
        return std::end(value_);
    }

    [[nodiscard]]
    auto begin() -> container_t::iterator
    {
        return std::begin(value_);
    }

    [[nodiscard]]
    auto end() -> container_t::iterator
    {
        return std::end(value_);
    }

    constexpr auto operator<=>(physical_vector const&) const = default;

public:
    container_t value_;
};

template <std::size_t N, std::floating_point F>
auto operator<<(std::ostream& os, physical_vector<N, F> const& pv) noexcept
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
