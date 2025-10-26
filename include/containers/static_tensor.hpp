#ifndef AMR_INCLUDED_STATIC_TENSOR
#define AMR_INCLUDED_STATIC_TENSOR

#include "container_concepts.hpp"
#include "multi_index.hpp"
#include "static_layout.hpp"
#include "utility/compile_time_utility.hpp"
#include "utility/utility_concepts.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <optional>
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
    constexpr static auto flat_size() noexcept -> size_type
    {
        return layout_t::flat_size();
    }

    [[nodiscard]]
    constexpr static auto elements() noexcept -> size_type
    {
        return layout_t::elements();
    }

    [[nodiscard]]
    constexpr static auto rank() noexcept -> rank_t
    {
        return layout_t::rank();
    }

    [[nodiscard]]
    constexpr static auto sizes() noexcept -> auto const&
    {
        return layout_t::sizes();
    }

    [[nodiscard]]
    constexpr static auto size(index_t const i) noexcept -> size_type
    {
        assert(i < rank());
        return layout_t::size(i);
    }

    [[nodiscard]]
    constexpr static auto strides() noexcept -> auto const&
    {
        return layout_t::strides();
    }

    [[nodiscard]]
    constexpr static auto stride(index_t const i) noexcept -> size_type
    {
        assert(i < rank());
        return layout_t::stride(i);
    }

    [[nodiscard]]
    constexpr static auto linear_index(index_t const (&idxs)[rank()]) noexcept -> index_t
    {
        return layout_t::linear_index(idxs);
    }

    [[nodiscard]]
    constexpr static auto linear_index(multi_index_t const& multi_idx) noexcept -> index_t
    {
        return layout_t::linear_index(multi_idx);
    }

public:
    [[nodiscard]]
    constexpr static auto zero() noexcept -> static_tensor
    {
        return static_tensor{};
    }

public:
    template <class... I>
        requires(sizeof...(I) == rank()) && (std::integral<std::remove_cvref_t<I>> && ...)
    [[nodiscard]]
    constexpr auto operator[](I&&... idxs) const noexcept -> const_reference
    {
        index_t indices[rank()] = { static_cast<index_t>(idxs)... };
        return data_[linear_index(indices)];
    }

    template <class... I>
        requires(sizeof...(I) == rank()) && (std::integral<std::remove_cvref_t<I>> && ...)
    [[nodiscard]]
    constexpr auto operator[](I&&... idxs) noexcept -> reference
    {
        return const_cast<reference>(
            std::as_const(*this).operator[](std::forward<decltype(idxs)>(idxs)...)
        );
    }

    [[nodiscard]]
    constexpr auto operator[](multi_index_t const& multi_idx) const noexcept
        -> const_reference
    {
        return data_[linear_index(multi_idx)];
    }

    [[nodiscard]]
    constexpr auto operator[](multi_index_t const& multi_idx) noexcept -> reference
    {
        return const_cast<reference>(std::as_const(*this).operator[](multi_idx));
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
        static constexpr auto pool = []
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
        static constexpr auto pool = []
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
        static constexpr auto pool = []
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
        width = std::clamp((int)std::ceil(std::log10(std::ranges::max(t))), 1, 7);
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
