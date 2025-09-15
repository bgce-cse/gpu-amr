#ifndef AMR_INCLUDED_MULTI_INDEX
#define AMR_INCLUDED_MULTI_INDEX

#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <iostream>
#include <type_traits>
#include <utility>

namespace amr::containers::index
{

template <std::integral Index_Type, std::integral auto N, std::integral auto... Ns>
struct static_multi_index
{
public:
    using index_t         = Index_Type;
    using size_type       = std::common_type_t<decltype(N), decltype(Ns)...>;
    using rank_t          = size_type;
    using const_iterator  = index_t const*;
    using iterator        = index_t*;
    using const_reference = index_t const&;
    using reference       = index_t&;

    struct increment_result_t
    {
        constexpr auto incremented_idx() const noexcept -> rank_t
        {
            return incremented_idx_;
        }

        constexpr auto reverse_incremented_idx() const noexcept -> rank_t
        {
            return rank_t{ s_rank - incremented_idx_ } - 1;
        }

        constexpr operator bool() const noexcept
        {
            return incremented_idx_ != s_rank;
        }

        constexpr auto is_fastest() const noexcept -> bool
        {
            return incremented_idx_ == rank_t{ s_rank - 1 };
        }

        rank_t incremented_idx_;
    };

    inline static constexpr size_type                     s_rank  = sizeof...(Ns) + 1;
    inline static constexpr std::array<size_type, s_rank> s_sizes = { N, Ns... };

    constexpr auto reset() noexcept -> void
    {
        std::ranges::fill(value_, index_t{});
    }

    [[nodiscard]]
    constexpr auto increment() noexcept -> increment_result_t
    {
        for (rank_t d = s_rank; d-- > 0;)
        {
            assert(value_[d] >= 0 && value_[d] < s_sizes[d]);
            if (value_[d] != s_sizes[d] - 1)
            {
                ++value_[d];
                return { d };
            }
            else
            {
                value_[d] = index_t{};
            }
        }
        return { s_rank };
    }

    [[nodiscard]]
    constexpr auto operator[](index_t const i) noexcept -> reference
    {
        return const_cast<reference>(std::as_const(*this).operator[](i));
    }

    [[nodiscard]]
    constexpr auto operator[](index_t const i) const noexcept -> const_reference
    {
        return value_[i];
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

    [[nodiscard]]
    constexpr auto operator<=>(static_multi_index const&) const noexcept = default;

private:
    std::array<index_t, s_rank> value_;
};

template <std::integral Index_Type, std::integral auto N, std::integral auto... Ns>
auto operator<<(
    std::ostream&                                   os,
    static_multi_index<Index_Type, N, Ns...> const& idx
) noexcept -> std::ostream&
{
    using idx_t = std::remove_cvref_t<decltype(idx)>;
    os << "{ ";
    for (typename idx_t::index_t d{}; d != idx_t::s_rank;)
    {
        os << idx[d];
        os << (++d != idx_t::s_rank ? ", " : " ");
    }
    return os << '}';
}

} // namespace amr::containers::index

#endif // AMR_INCLUDED_MULTI_INDEX
