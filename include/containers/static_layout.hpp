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

template <std::integral auto N, std::integral auto... Ns>
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
    using multi_index_t   = index::static_multi_index<index_t, N, Ns...>;

    inline static constexpr rank_t                        s_rank = sizeof...(Ns) + 1;
    inline static constexpr std::array<size_type, s_rank> s_sizes =
        multi_index_t::s_sizes;
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
    inline static constexpr size_type s_flat_size = (N * ... * Ns);

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
