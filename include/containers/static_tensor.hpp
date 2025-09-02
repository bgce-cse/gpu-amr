#ifndef AMR_INLUDED_STATIC_TENSOR
#define AMR_INLUDED_STATIC_TENSOR

#include "utility/utility_concepts.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <type_traits>

#ifndef NDEBUG
#    define AMR_CONTANERS_CHECKBOUNDS
#endif

namespace amr::containers
{

template <typename T, std::integral auto N, std::integral auto... Ns>
    requires utility::concepts::are_same<decltype(N), decltype(Ns)...> && (N > 0) &&
             ((Ns > 0) && ...)
class static_tensor
{
public:
    using value_type      = std::remove_cv_t<T>;
    using size_type       = std::common_type_t<decltype(N), decltype(Ns)...>;
    using index_t         = size_type;
    using const_iterator  = value_type const*;
    using iterator        = value_type*;
    using const_reference = value_type const&;
    using reference       = value_type&;

    inline static constexpr size_type s_rank  = sizeof...(Ns) + 1;
    inline static constexpr auto      s_sizes = std::array<size_type, s_rank>{
        { N, Ns... }
    };
    inline static constexpr auto s_strides = []
    {
        std::array<size_type, s_rank> strides{};
        strides[s_rank - 1] = size_type{ 1 };
        for (size_type d = s_rank - 1; d != 0; --d)
        {
            strides[d - 1] = strides[d] * s_sizes[d];
        }
        return strides;
    }();
    inline static constexpr size_type s_flat_size =
        std::ranges::fold_left(s_sizes, size_type{ 1 }, std::multiplies{});

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
    constexpr static auto rank() noexcept -> size_type
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

    template <class... I>
        requires(sizeof...(I) == s_rank) &&
                (std::is_same_v<std::remove_cvref_t<I>, index_t> && ...)
    [[nodiscard]] constexpr static auto linear_index(I&&... idxs) noexcept -> index_t
    {
        index_t idx_v[s_rank] = { static_cast<index_t>(idxs)... };
        auto    linear_idx    = index_t{};
        for (auto d = size_type{}; d < s_rank; ++d)
        {
            linear_idx += idx_v[d] * s_strides[d];
        }
        return linear_idx;
    }

public:
    template <class... I>
        requires(sizeof...(I) == rank()) && (std::integral<std::remove_cvref_t<I>> && ...)
    [[nodiscard]]
    constexpr auto operator[](I&&... idxs) const noexcept -> const_reference
    {
        return data_[linear_index(static_cast<index_t>(idxs)...)];
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
#ifdef AMR_CONTANERS_CHECKBOUNDS
    auto assert_in_bounds(index_t const (&idxs)[s_rank]) const noexcept -> void
    {
        for (auto d = size_type{}; d != s_rank; ++d)
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

} // namespace amr::containers

#endif // AMR_INCLUDED_STATIC_TENSOR
