#ifndef AMR_INCLUDED_STATIC_TENSOR
#define AMR_INCLUDED_STATIC_TENSOR

#include "container_concepts.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <optional>
#include <ranges>
#include <string_view>
#include <type_traits>

namespace amr::containers
{

template <typename T, concepts::StaticLayout Layout>
class static_tensor
{
public:
    using value_type      = std::remove_cv_t<T>;
    using layout_t        = Layout;
    using shape_t         = typename layout_t::shape_t;
    using size_type       = typename layout_t::size_type;
    using index_t         = typename layout_t::index_t;
    using rank_t          = typename layout_t::rank_t;
    using multi_index_t   = typename layout_t::multi_index_t;
    using const_iterator  = value_type const*;
    using iterator        = value_type*;
    using const_reference = value_type const&;
    using reference       = value_type&;

private:
    static_assert(std::is_trivially_copyable_v<T>);
    static_assert(std::is_standard_layout_v<T>);

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
    static constexpr auto sizes() noexcept -> auto const&
    {
        return layout_t::sizes();
    }

    [[nodiscard]]
    static constexpr auto size(index_t const i) noexcept -> size_type
    {
        assert(i < rank());
        return layout_t::size(i);
    }

    [[nodiscard]]
    static constexpr auto strides() noexcept -> auto const&
    {
        return layout_t::strides();
    }

    [[nodiscard]]
    static constexpr auto stride(index_t const i) noexcept -> size_type
    {
        assert(i < rank());
        return layout_t::stride(i);
    }

    template <std::integral... I>
        requires(sizeof...(I) == rank())
    [[nodiscard]]
    static constexpr auto linear_index(I&&... idxs) noexcept -> index_t
    {
        return layout_t::linear_index(std::forward<decltype(idxs)>(idxs)...);
    }

    [[nodiscard]]
    static constexpr auto
        linear_index(std::ranges::contiguous_range auto const& idxs) noexcept -> index_t
        requires(std::ranges::size(idxs) == rank())
    {
        return layout_t::linear_index(idxs);
    }

public:
    [[nodiscard]]
    static constexpr auto zero() noexcept -> static_tensor
    {
        return static_tensor{};
    }

public:
    [[nodiscard]]
    constexpr auto
        operator[](std::ranges::contiguous_range auto const& idxs) const noexcept
        -> const_reference
        requires(std::ranges::size(idxs) == rank() && std::is_same_v<std::ranges::range_value_t<decltype(idxs)>, index_t>)
    {
        return data_[linear_index(idxs)];
    }

    [[nodiscard]]
    constexpr auto operator[](std::ranges::contiguous_range auto const& idxs) noexcept
        -> reference
    {
        return const_cast<reference>(std::as_const(*this).operator[](idxs));
    }

    template <typename... I>
        requires(sizeof...(I) == rank()) && (std::integral<std::remove_cvref_t<I>> && ...)
    [[nodiscard]]
    constexpr auto operator[](I const&... idxs) const noexcept -> const_reference
    {
        return data_[linear_index(static_cast<index_t>(idxs)...)];
    }

    template <typename... I>
        requires(sizeof...(I) == rank()) && (std::integral<std::remove_cvref_t<I>> && ...)
    [[nodiscard]]
    constexpr auto operator[](I const&... idxs) noexcept -> reference
    {
        return const_cast<reference>(std::as_const(*this).operator[](idxs...));
    }

    [[nodiscard]]
    constexpr auto operator[](index_t const linear_idx) const noexcept -> const_reference
    {
        return data_[linear_idx];
    }

    [[nodiscard]]
    constexpr auto operator[](index_t const linear_idx) noexcept -> reference
    {
        return const_cast<reference>(std::as_const(*this).operator[](linear_idx));
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

private:
    // TODO: Alignment?
    value_type data_[flat_size()];
};

template <typename T, concepts::StaticLayout Layout>
auto operator<<(std::ostream& os, static_tensor<T, Layout> const& t) noexcept
    -> std::ostream&
{
    using tensor_t                 = std::remove_cvref_t<decltype(t)>;
    static constexpr auto rank     = tensor_t::rank();
    static constexpr auto newlines = []
    {
        static constexpr auto pool = [] constexpr -> auto
        {
            std::array<char, rank> arr{};
            for (auto& e : arr)
            {
                e = '\n';
            }
            return arr;
        }();
        std::array<std::string_view, rank> arr{};
        for (std::size_t d = 0; d != rank; ++d)
        {
            arr[d] = std::string_view(pool.data(), d + 1);
        }
        return arr;
    }();
    static constexpr auto prefixes = []
    {
        static constexpr auto pool = [] constexpr -> auto
        {
            std::array<char, rank * 2> arr{};
            for (std::size_t d = 0; d != rank; ++d)
            {
                arr[d]        = ' ';
                arr[d + rank] = '[';
            }
            return arr;
        }();
        std::array<std::string_view, rank> arr{};
        for (std::size_t d = 0; d != rank; ++d)
        {
            arr[d] = std::string_view(pool.data() + d + 1, rank);
        }
        return arr;
    }();
    static constexpr auto postfix = []
    {
        static constexpr auto pool = [] constexpr -> auto
        {
            std::array<char, rank> arr{};
            for (auto& e : arr)
            {
                e = ']';
            }
            return arr;
        }();
        std::array<std::string_view, rank> arr{};
        for (std::size_t d = 0; d != rank; ++d)
        {
            arr[d] = std::string_view(pool.data(), d + 1);
        }
        return arr;
    }();

    auto multi_idx = typename tensor_t::multi_index_t{};

    std::optional<int> width = std::nullopt;
    if constexpr (std::is_arithmetic_v<T>)
    {
        // TODO: Improve. Max is not necessarily the most restrictive value
        width = std::clamp((int)std::ceil(std::log10(std::ranges::max(t))), 1, 7) + 1;
    }

    os << prefixes[rank - 1];
    while (true)
    {
        if constexpr (std::is_arithmetic_v<T>)
        {
            os << std::setw(width.value()) << std::setfill(' ');
        }
        os << t[multi_idx];
        auto res = multi_idx.increment();
        if (!res)
        {
            break;
        }
        if (res.is_fastest())
        {
            os << ", ";
        }
        else
        {
            const auto i = res.reverse_incremented_idx() - 1;
            os << postfix[i];
            os << newlines[i];
            os << prefixes[i];
        }
    }
    os << postfix[rank - 1];
    return os;
}

} // namespace amr::containers

#endif // AMR_INCLUDED_STATIC_TENSOR
