#ifndef AMR_INCLUDED_STATIC_VECTOR
#define AMR_INCLUDED_STATIC_VECTOR

#include "static_layout.hpp"
#include "static_shape.hpp"
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
    using shape_t         = static_shape<N>;
    using layout_t        = static_layout<shape_t>;
    using size_type       = typename layout_t::size_type;
    using size_pack_t     = typename layout_t::size_pack_t;
    using index_t         = typename layout_t::index_t;
    using rank_t          = typename layout_t::rank_t;
    using const_iterator  = value_type const*;
    using iterator        = value_type*;
    using const_reference = value_type const&;
    using reference       = value_type&;

private:
    static_assert(std::is_trivially_copyable_v<value_type>);
    static_assert(std::is_standard_layout_v<value_type>);

public:
    [[nodiscard]]
    static constexpr auto flat_size() noexcept -> size_type
    {
        return layout_t::flat_size();
    }

    [[nodiscard]]
    static constexpr auto elements() noexcept -> size_type
    {
        return layout_t::elements();
    }

    [[nodiscard]]
    static constexpr auto rank() noexcept -> rank_t
    {
        return layout_t::rank();
    }

    [[nodiscard]]
    static constexpr auto size(index_t const i) noexcept -> size_type
    {
        assert(i < rank());
        return layout_t::size(i);
    }

    [[nodiscard]]
    static constexpr auto sizes() noexcept -> auto const&
    {
        return layout_t::sizes();
    }

    [[nodiscard]]
    static constexpr auto strides() noexcept -> auto const&
    {
        return layout_t::strides;
    }

    [[nodiscard]]
    static constexpr auto stride(index_t const i) noexcept -> size_type
    {
        assert(i < rank());
        return layout_t::stride(i);
    }

    [[nodiscard]]
    static constexpr auto
        linear_index(std::ranges::contiguous_range auto const& idxs) noexcept -> index_t

    {
        // TODO
        // assert(std::ranges::size(idxs) == rank());
        return layout_t::linear_index(idxs);
    }

public:
    [[nodiscard]]
    constexpr auto
        operator[](std::ranges::contiguous_range auto const& idxs) const noexcept
        -> const_reference
    // requires(std::ranges::size(idxs) == rank() &&
    // std::is_same_v<std::ranges::range_value_t<decltype(idxs)>, index_t>) #TODO: @Miguel
    {
        return data_[linear_index(idxs)];
    }

    [[nodiscard]]
    constexpr auto operator[](std::ranges::contiguous_range auto const& idxs) noexcept
        -> reference
    {
        return const_cast<reference>(std::as_const(*this).operator[](idxs));
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

#if __GNUC__ >= 14
    [[nodiscard]]
    constexpr auto front(this auto&& self) noexcept -> decltype(auto)
    {
        return std::forward<decltype(self)>(self).data_.front();
    }

    [[nodiscard]]
    constexpr auto back(this auto&& self) noexcept -> decltype(auto)
    {
        return std::forward<decltype(self)>(self).data_.front();
    }
#else
    [[nodiscard]]
    constexpr auto front() const noexcept -> const_reference
    {
        return data_.front();
    }

    [[nodiscard]]
    constexpr auto front() noexcept -> reference
    {
        return data_.front();
    }

    [[nodiscard]]
    constexpr auto back() const noexcept -> const_reference
    {
        return data_.back();
    }

    [[nodiscard]]
    constexpr auto back() noexcept -> reference
    {
        return data_.back();
    }
#endif

#ifdef AMR_CONTAINERS_CHECKBOUNDS
    constexpr auto assert_in_bounds(size_type const idx) const noexcept -> void
    {
        assert(idx < elements());
        if constexpr (std::is_signed_v<size_type>)
        {
            assert(idx >= size_type{});
        }
    }
#endif

    [[nodiscard]]
    constexpr auto operator<=>(static_vector const&) const noexcept = default;

public:
    value_type data_[flat_size()];
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
