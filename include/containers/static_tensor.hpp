#ifndef AMR_INCLUDED_STATIC_TENSOR
#define AMR_INCLUDED_STATIC_TENSOR

#include "multi_index.hpp"
#include "utility/compile_time_utility.hpp"
#include "utility/utility_concepts.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <functional>
#include <iomanip>
#include <string>
#include <type_traits>
#include <utility>

#ifndef NDEBUG
#    define AMR_CONTAINERS_CHECKBOUNDS
#endif

namespace amr::containers
{

template <typename T, std::integral auto N, std::integral auto... Ns>
    requires utility::concepts::are_same<decltype(N), decltype(Ns)...> && (N > 0) &&
             ((Ns > 0) && ...)
class static_tensor
{
public:
    using value_type = std::remove_cv_t<T>;
    // TODO: This can be dangerous, maybe hardcode a type once we know what we
    // need
    using size_type       = std::common_type_t<decltype(N), decltype(Ns)...>;
    using index_t         = size_type;
    using rank_t          = size_type;
    using const_iterator  = value_type const*;
    using iterator        = value_type*;
    using const_reference = value_type const&;
    using reference       = value_type&;
    using multi_index_t   = index::static_multi_index<index_t, N, Ns...>;

    inline static constexpr rank_t                        s_rank = sizeof...(Ns) + 1;
    inline static constexpr std::array<size_type, s_rank> s_sizes =
        multi_index_t::s_sizes;
    inline static constexpr std::array<size_type, s_rank> s_reverse_sizes =
        multi_index_t::s_reverse_sizes;
    inline static constexpr auto s_strides = []
    {
        std::array<size_type, s_rank> strides{};
        strides[0] = size_type{ 1 };
        for (rank_t d = 1; d != s_rank; ++d)
        {
            strides[d] = strides[d - 1] * s_reverse_sizes[d - 1];
        }
        return strides;
    }();
    inline static constexpr auto s_reverse_strides = []
    {
        auto strides = s_strides;
        std::ranges::reverse(strides);
        return strides;
    }();
    inline static constexpr size_type s_flat_size = (N * ... * Ns);

    static_assert(std::is_trivially_copyable_v<T>);
    static_assert(std::is_standard_layout_v<T>);
    static_assert(s_rank > 0);

public:
    [[nodiscard]]
    constexpr static auto flat_size() noexcept -> size_type
    {
        return s_flat_size;
    }

    [[nodiscard]]
    constexpr static auto rank() noexcept -> rank_t
    {
        return s_rank;
    }

    [[nodiscard]]
    constexpr static auto size(index_t const i) noexcept -> size_type
    {
        assert(i < s_rank);
        return s_sizes[i];
    }

    [[nodiscard]]
    constexpr static auto stride(index_t const i) noexcept -> size_type
    {
        assert(i < s_rank);
        return s_strides[i];
    }

    [[nodiscard]]
    constexpr static auto linear_index(index_t const (&idxs)[s_rank]) noexcept -> index_t
    {
#ifdef AMR_CONTAINERS_CHECKBOUNDS
        assert_in_bounds(idxs);
#endif
        auto linear_idx = index_t{};
        for (auto d = rank_t{}; d != s_rank; ++d)
        {
            linear_idx += idxs[d] * s_reverse_strides[d];
        }
        assert(linear_idx < flat_size());
        return linear_idx;
    }

    [[nodiscard]]
    constexpr static auto linear_index(multi_index_t const& multi_idx) noexcept -> index_t
    {
        auto linear_idx = index_t{};
        for (auto d = rank_t{}; d != s_rank; ++d)
        {
            linear_idx += multi_idx.get(d) * s_strides[d];
        }
        assert(linear_idx < flat_size());
        return linear_idx;
    }

public:
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
        index_t indices[s_rank] = { static_cast<index_t>(idxs)... };
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

    constexpr auto operator[](multi_index_t const& multi_idx) const noexcept
        -> const_reference
    {
        return data_[linear_index(multi_idx)];
    }

    constexpr auto operator[](multi_index_t const& multi_idx) noexcept -> reference
    {
        return const_cast<reference>(std::as_const(*this).operator[](multi_idx));
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
#ifdef AMR_CONTAINERS_CHECKBOUNDS
    static auto assert_in_bounds(index_t const (&idxs)[s_rank]) noexcept -> void
    {
        for (auto d = rank_t{}; d != s_rank; ++d)
        {
            assert(idxs[d] < s_sizes[d]);
            if constexpr (std::is_signed_v<index_t>)
            {
                assert(idxs[d] >= 0);
            }
        }
    }
#endif

private:
    // TODO: Alignment?
    value_type data_[s_flat_size];
};

template <typename T, std::integral auto N, std::integral auto... Ns>
auto operator<<(std::ostream& os, static_tensor<T, N, Ns...> const& t) noexcept
    -> std::ostream&
{
    using tensor_t                 = std::remove_cvref_t<decltype(t)>;
    static constexpr auto rank     = tensor_t::s_rank;
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

    auto       multi_idx = typename tensor_t::multi_index_t{};
    const auto w = std::clamp((int)std::ceil(std::log10(std::ranges::max(t))), 1, 7);

    os << prefixes[rank - 1];
    while (true)
    {
        os << std::setw(w) << std::setfill(' ') << t[multi_idx];
        auto res = multi_idx.increment();
        if (!res)
        {
            break;
        }
        if (res.incremented_idx == 0)
        {
            os << ", ";
        }
        else
        {
            const auto i = res.incremented_idx - 1;
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
