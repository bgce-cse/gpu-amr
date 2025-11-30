#ifndef AMR_INCLUDED_STATIC_LAYOUT
#define AMR_INCLUDED_STATIC_LAYOUT

#include "container_concepts.hpp"
#include "multi_index.hpp"
#include <array>
#include <cassert>
#include <numeric>
#include <utility>

#ifndef NDEBUG
#    define AMR_CONTAINERS_CHECKBOUNDS
#endif

namespace amr::containers
{

template <concepts::StaticShape Shape>
class static_layout
{
public:
    using shape_t       = Shape;
    using size_type     = typename shape_t::size_type;
    using rank_t        = typename shape_t::rank_t;
    using index_t       = typename shape_t::index_t;
    using multi_index_t = index::static_multi_index<index_t, shape_t>;

private:
    inline static constexpr rank_t s_rank    = shape_t::rank();
    inline static constexpr auto&  s_sizes   = shape_t::sizes();
    inline static constexpr auto   s_strides = []
    {
        std::remove_cvref_t<decltype(s_sizes)> strides{};
        strides[s_rank - 1] = size_type{ 1 };
        for (rank_t d = s_rank - 1; d-- > 0;)
        {
            strides[d] = strides[d + 1] * s_sizes[d + 1];
        }
        return strides;
    }();
    inline static constexpr size_type s_flat_size = shape_t::elements();
    static_assert(s_rank > 0);
    static_assert(s_flat_size >= shape_t::elements());

public:
    [[nodiscard]]
    constexpr static auto flat_size() noexcept -> size_type
    {
        return s_flat_size;
    }

    [[nodiscard]]
    constexpr static auto elements() noexcept -> size_type
    {
        return shape_t::elements();
    }

    [[nodiscard]]
    constexpr static auto rank() noexcept -> rank_t
    {
        return s_rank;
    }

    [[nodiscard]]
    constexpr static auto sizes() noexcept -> auto const&
    {
        return shape_t::sizes();
    }

    [[nodiscard]]
    constexpr static auto size(index_t const i) noexcept -> size_type
    {
        return shape_t::size(i);
    }

    [[nodiscard]]
    constexpr static auto strides() noexcept -> auto const&
    {
        return s_strides;
    }

    [[nodiscard]]
    constexpr static auto stride(index_t const i) noexcept -> size_type
    {
#ifdef AMR_CONTAINERS_CHECKBOUNDS
        assert_in_bounds(i);
#endif
        return s_strides[i];
    }

    template <std::integral... I>
        requires(sizeof...(I) == rank())
    [[nodiscard]]
    constexpr static auto linear_index(I&&... idxs) noexcept -> index_t
    {
#ifdef AMR_CONTAINERS_CHECKBOUNDS
        const index_t vidxs[rank()]{ static_cast<index_t>(idxs)... };
        assert_in_bounds(vidxs);
#endif
        const auto linear_idx = []<std::size_t... Indices>(
                                    std::index_sequence<Indices...>, auto&&... index_pack
                                )
        {
            return ((index_pack * s_strides[Indices]) + ...);
        }(std::make_index_sequence<sizeof...(I)>{}, std::forward<I>(idxs)...);

        assert(linear_idx < elements());
        if constexpr (std::is_signed_v<index_t>)
        {
            assert(linear_idx >= 0);
        }
        return linear_idx;
    }

    [[nodiscard]]
    constexpr static auto
        linear_index(std::ranges::contiguous_range auto const& idxs) noexcept
        -> index_t
    {
        auto linear_idx = std::transform_reduce(
            std::cbegin(idxs),
            std::cend(idxs),
            std::cbegin(s_strides),
            index_t{}
        );
        assert(linear_idx < elements());
        if constexpr (std::is_signed_v<index_t>)
        {
            assert(linear_idx >= 0);
        }
        return linear_idx;
    }

    [[nodiscard]]
    constexpr static auto multi_index(index_t linear_idx) noexcept -> multi_index_t
    {
#ifdef AMR_CONTAINERS_CHECKBOUNDS
        assert_in_bounds(linear_idx);
#endif
        multi_index_t ret{};
        for (rank_t d = 0; d != s_rank; ++d)
        {
            ret[d] = linear_idx / s_strides[d];
            linear_idx %= s_strides[d];
        }
        return ret;
    }

private:
#ifdef AMR_CONTAINERS_CHECKBOUNDS
    static auto assert_in_bounds(index_t const (&idxs)[s_rank]) noexcept -> void
        requires(s_rank != 1)
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

    constexpr static auto assert_in_bounds(index_t idx) noexcept -> void
    {
        if (!std::is_constant_evaluated())
        {
            assert(idx < s_flat_size);
            if constexpr (std::is_signed_v<index_t>)
            {
                assert(idx >= 0);
            }
        }
    }
#endif
};

} // namespace amr::containers

#endif // AMR_INCLUDED_STATIC_LAYOUT
