#pragma once

#include <array>
#include <cstddef>

template <std::size_t... Dims>
class PatchIndexer
{
public:
    static constexpr std::size_t                N    = sizeof...(Dims);
    static constexpr std::array<size_t, N> s_dims = { Dims... };

    template <size_t Fanout>
    static constexpr auto check_dims_multiple_of_fanout() noexcept -> bool
    {
        return ((Dims % Fanout == 0) && ...);
    }

    static constexpr std::array<size_t, N> compute_strides()
    {
        std::array<size_t, N> strides{};
        strides[N - 1] = 1;
        for (int i = N - 2; i >= 0; --i)
        {
            strides[i] = strides[i + 1] * s_dims[i + 1];
        }
        return strides;
    }

    static constexpr std::array<size_t, N> strides = compute_strides();

    static constexpr std::size_t flat_index(const std::array<size_t, N>& indices)
    {
        std::size_t idx = 0;
        for (size_t d = 0; d < N; ++d)
        {
            idx += indices[d] * strides[d];
        }
        return idx;
    }

    static constexpr std::array<size_t, N> to_multi_dim(size_t flat_idx)
    {
        std::array<size_t, N> indices{};
        for (size_t d = 0; d < N; ++d)
        {
            indices[d] = flat_idx / strides[d];
            flat_idx %= strides[d];
        }
        return indices;
    }

    static constexpr std::size_t total_size()
    {
        std::size_t prod = 1;
        for (size_t d : s_dims)
            prod *= d;
        return prod;
    }

    static constexpr std::size_t dim_size(size_t i)
    {
        return s_dims[i];
    }
};
