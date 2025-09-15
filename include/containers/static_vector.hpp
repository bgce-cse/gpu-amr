#ifndef AMR_INCLUDED_STATIC_VECTOR
#define AMR_INCLUDED_STATIC_VECTOR

#include "utility/casts.hpp"
#include <array>
#include <concepts>
#include <iomanip>
#include <ios>
#include <iterator>
#include <ostream>
#include <type_traits>

#ifndef NDEBUG
#    define AMR_CONTAINERS_CHECKBOUNDS
#endif

namespace amr::containers
{

template <typename T, std::integral auto N>
struct static_vector
{
    using value_type      = std::remove_cv_t<T>;
    using size_type       = std::size_t;
    using const_iterator  = value_type const*;
    using iterator        = value_type*;
    using const_reference = value_type const&;
    using reference       = value_type&;

    // TODO: Alignment
    static constexpr auto s_size = N;

    static_assert(std::is_trivially_copyable_v<T>);
    static_assert(std::is_standard_layout_v<T>);

    [[nodiscard]]
    constexpr static auto size() noexcept -> size_type
    {
        return s_size;
    }

    [[nodiscard]]
    constexpr auto operator[](size_type const idx) const noexcept -> const_reference
    {
#ifdef AMR_CONTAINERS_CHECKBOUNDS
        assert_in_bounds(idx);
#endif
        return data_[idx];
    }

    [[nodiscard]]
    constexpr auto operator[](size_type const idx) noexcept -> reference
    {
        return const_cast<T&>(std::as_const(*this).operator[](idx));
    }

    [[nodiscard]]
    constexpr auto cbegin() const noexcept -> const_iterator
    {
        return std::cbegin(data_);
    }

    [[nodiscard]]
    constexpr auto cend() const noexcept -> const_iterator
    {
        return std::cend(data_);
    }

    [[nodiscard]]
    constexpr auto begin() const noexcept -> const_iterator
    {
        return std::begin(data_);
    }

    [[nodiscard]]
    constexpr auto end() const noexcept -> const_iterator
    {
        return std::end(data_);
    }

    [[nodiscard]]
    constexpr auto begin() noexcept -> iterator
    {
        return std::begin(data_);
    }

    [[nodiscard]]
    constexpr auto end() noexcept -> iterator
    {
        return std::end(data_);
    }

#ifdef AMR_CONTAINERS_CHECKBOUNDS
    constexpr auto assert_in_bounds(size_type const idx) const noexcept -> void
    {
        assert(idx < s_size);
        if constexpr (std::is_signed_v<size_type>)
        {
            assert(idx >= size_type{});
        }
    }
#endif

    constexpr auto operator<=>(static_vector const&) const noexcept = default;

public:
    value_type data_[s_size];
};

template <typename Value_Type, std::integral auto N>
auto operator<<(std::ostream& os, static_vector<Value_Type, N> const& v) noexcept
    -> std::ostream&
{
    using vector_t = static_vector<Value_Type, N>;
    if constexpr (std::is_floating_point_v<typename vector_t::value_type>)
    {
        os << std::fixed;
        os << std::setprecision(4);
    }
    os << "{ ";
    std::size_t n{ 0 };
    for (auto const& e : v)
    {
        os << e << (++n != N ? ", " : " ");
    }
    os << '}';
    if constexpr (std::is_floating_point_v<typename vector_t::value_type>)
    {
        os << std::defaultfloat;
    }
    return os;
}

} // namespace amr::containers

#endif // AMR_INCLUDED_STATIC_VECTOR
