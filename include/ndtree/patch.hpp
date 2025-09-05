#pragma once

#include <array>
#include <cstddef>
#include <numeric>
#include <type_traits>

template <size_t... Dims>
class PatchIndexer {
public:
    static constexpr size_t N = sizeof...(Dims);
    static constexpr std::array<size_t, N> dims = {Dims...};

    template <size_t Fanout>
static constexpr bool check_dims_multiple_of_fanout() {
    return ((Dims % Fanout == 0) && ...);
}

    static constexpr std::array<size_t, N> compute_strides() {
        std::array<size_t, N> strides{};
        strides[N-1] = 1;
        for (int i = N-2; i >= 0; --i) {
            strides[i] = strides[i+1] * dims[i+1];
        }
        return strides;
    }

    static constexpr std::array<size_t, N> strides = compute_strides();


    static constexpr size_t flat_index(const std::array<size_t, N>& indices) {
        size_t idx = 0;
        for (size_t d = 0; d < N; ++d) {
            idx += indices[d] * strides[d];
        }
        return idx;
    }


    static constexpr std::array<size_t, N> to_multi_dim(size_t flat_idx) {
        std::array<size_t, N> indices{};
        for (size_t d = 0; d < N; ++d) {
            indices[d] = flat_idx / strides[d];
            flat_idx %= strides[d];
        }
        return indices;
    }


    static constexpr size_t total_size() {
        size_t prod = 1;
        for (size_t d : dims) prod *= d;
        return prod;
    }


    static constexpr size_t dim_size(size_t i) { return dims[i]; }
};