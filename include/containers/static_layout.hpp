#ifndef AMR_INCLUDED_STATIC_SHAPE
#define AMR_INCLUDED_STATIC_SHAPE

#include "utility/compile_time_utility.hpp"
#include "utility/utility_concepts.hpp"
#include <array>
#include <cassert>
#include <numeric>

#ifndef NDEBUG
#    define AMR_CONTAINERS_CHECKBOUNDS
#endif

namespace amr::containers
{

template <std::integral auto H, std::integral auto N, std::integral auto... Ns>
    requires utility::concepts::are_same<decltype(N), decltype(Ns)...> && (N > 0) &&
             ((Ns > 0) && ...)
class static_layout
{
public:
    // TODO: This can be dangerous, maybe hardcode a type once we know what we
    // need
    using size_type       = std::common_type_t<decltype(N), decltype(Ns)...>;
    using index_t         = size_type;
    using rank_t          = size_type;
    using logical_multi_index_t   = index::static_multi_index<index_t, N, Ns...>;
    using multi_index_t   = index::static_multi_index<index_t, N + 2*H, (Ns + 2*H)...>;


    inline static constexpr auto s_halo = H; 
    inline static constexpr rank_t                        s_rank = sizeof...(Ns) + 1;

    inline static constexpr std::array<size_type, s_rank> s_logical_sizes =
        logical_multi_index_t::s_sizes;
    inline static constexpr std::array<size_type, s_rank> s_sizes =
    multi_index_t::s_sizes;

    inline static constexpr auto s_logical_strides = []
    {
        std::array<size_type, s_rank> strides{};
        strides[s_rank - 1] = size_type{ 1 };
        for (rank_t d = s_rank - 1; d-- > 0;)
        {
            strides[d] = strides[d + 1] * s_logical_sizes[d + 1];
        }
        return strides;
    }();


    inline static constexpr auto s_strides = []
    {
        std::array<size_type, s_rank> strides{};
        strides[s_rank - 1] = size_type{ 1 };
        for (rank_t d = s_rank - 1; d-- > 0;)
        {
            strides[d] = strides[d + 1] * s_sizes[d + 1];
        }
        return strides;
    }();

    inline static constexpr size_type s_logical_flat_size = (N * ... * Ns);

    inline static constexpr size_type s_flat_size = ((N+2*H) * ... * (Ns+2*H));

    static_assert(s_rank > 0);

public:
    [[nodiscard]]
    constexpr static auto flat_logical_size() noexcept -> size_type
    {
        return s_logical_flat_size;
    }

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
    constexpr static auto logical_size(index_t const i) noexcept -> size_type
    {
        assert(i < s_rank);
        return s_logical_sizes[i];
    }

    [[nodiscard]]
    constexpr static auto stride(index_t const i) noexcept -> size_type
    {
        assert(i < s_rank);
        return s_strides[i];
    }

    [[nodiscard]]
    constexpr static auto logical_stride(index_t const i) noexcept -> size_type
    {
        assert(i < s_rank);
        return s_logical_strides[i];
    }


    [[nodiscard]]
    constexpr static auto linear_index(index_t const (&idxs)[s_rank]) noexcept -> index_t
    {
#ifdef AMR_CONTAINERS_CHECKBOUNDS
        assert_in_bounds(idxs);
#endif
        auto linear_idx = std::transform_reduce(
            std::cbegin(idxs), std::cend(idxs), std::cbegin(s_strides), index_t{}
        );
        assert(linear_idx < flat_size());
        if constexpr (std::is_signed_v<index_t>)
        {
            assert(linear_idx >= 0);
        }
        return linear_idx;
    }

    [[nodiscard]]
    constexpr static auto linear_index(multi_index_t const& multi_idx) noexcept -> index_t
    {
        auto linear_idx = std::transform_reduce(
            std::cbegin(multi_idx),
            std::cend(multi_idx),
            std::cbegin(s_strides),
            index_t{}
        );
        assert(linear_idx < flat_size());
        if constexpr (std::is_signed_v<index_t>)
        {
            assert(linear_idx >= 0);
        }
        return linear_idx;
    }



    [[nodiscard]]
constexpr static auto multi_index(size_t linear_index) noexcept -> multi_index_t
{
    assert(linear_index < s_flat_size);
    if constexpr (std::is_signed_v<index_t>)
        assert(linear_index >= 0);

    multi_index_t multi_idx{};
    auto remainder = linear_index;
    
    // Decode linear index back to multi-dimensional coordinates
    for (rank_t d = 0; d < s_rank; ++d)
    {
        auto stride = s_strides[d];
        multi_idx[d] = remainder / stride;
        remainder %= stride;
    }
    
    return multi_idx;
}


    [[nodiscard]]
    constexpr static auto linear_logical_index(index_t const (&idxs)[s_rank]) noexcept -> index_t
    {
#ifdef AMR_CONTAINERS_CHECKBOUNDS
        assert_in_bounds(idxs);
#endif
        auto linear_idx = std::transform_reduce(
            std::cbegin(idxs), std::cend(idxs), std::cbegin(s_logical_strides), index_t{}
        );
        assert(linear_idx < flat_size());
        if constexpr (std::is_signed_v<index_t>)
        {
            assert(linear_idx >= 0);
        }
        return linear_idx;
    }

    [[nodiscard]]
    constexpr static auto linear_logical_index(logical_multi_index_t const& multi_idx) noexcept -> index_t
    {
        auto linear_idx = std::transform_reduce(
            std::cbegin(multi_idx),
            std::cend(multi_idx),
            std::cbegin(s_logical_strides),
            index_t{}
        );
        assert(linear_idx < flat_size());
        if constexpr (std::is_signed_v<index_t>)
        {
            assert(linear_idx >= 0);
        }
        return linear_idx;
    }



   [[nodiscard]]
constexpr static auto logical_to_full_index(auto const linear_idx) noexcept -> index_t
{
    assert(linear_idx < s_logical_flat_size);
    if constexpr (std::is_signed_v<index_t>)
        assert(linear_idx >= 0);

    // 1. Decode logical coordinates
    std::array<index_t, s_rank> logical_coords{};
    auto remainder = linear_idx;
    for (rank_t d = 0; d < s_rank; ++d)
    {
        auto stride = s_logical_strides[d];
        logical_coords[d] = remainder / stride;
        remainder %= stride;
    }

    // 2. Shift by halo width
    for (auto& c : logical_coords)
        c += s_halo;

    // 3. Re-flatten using padded strides
    index_t full_index = 0;
    for (rank_t d = 0; d < s_rank; ++d)
        full_index += logical_coords[d] * s_strides[d];

    assert(full_index < s_flat_size);
    if constexpr (std::is_signed_v<index_t>)
        assert(full_index >= 0);

    return full_index;
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
};

} // namespace amr::containers

#endif // AMR_INCLUDED_STATIC_SHAPE
