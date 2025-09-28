#include "containers/container_utils.hpp"
#include "containers/static_layout.hpp"
#include "containers/static_tensor.hpp"
#include "utility/constexpr_functions.hpp"
#include <functional>
#include <iostream>

int main()
{
    constexpr auto N = 6;
    constexpr auto M = 8;
    constexpr auto K = 4;
    using index_t    = int;
    [[maybe_unused]]
    constexpr auto Fanout = 2;
    using tensor_t        = amr::containers::static_tensor<index_t, N, M, K>;
    using patch_shape_t   = amr::containers::utils::types::tensor::
        hypercube_t<tensor_t, Fanout, tensor_t::s_rank>;

    tensor_t from{};
    std::ranges::iota(from, 0);
    std::cout << from << '\n';
    for (auto const& e : from)
        std::cout << e << " ";
    std::cout << '\n';

    // -------------------------------------

    constexpr auto patch_index_to = []() constexpr
    {
        patch_shape_t to{};

        auto idx           = typename tensor_t::multi_index_t{};
        auto sized_strides = decltype(tensor_t::s_strides){};
        std::transform(
            std::cbegin(tensor_t::s_strides),
            std::cend(tensor_t::s_strides),
            std::cbegin(tensor_t::s_sizes),
            std::begin(sized_strides),
            std::multiplies{}
        );
        do
        {
            const auto offset = std::transform_reduce(
                std::cbegin(idx),
                std::cend(idx),
                std::cbegin(tensor_t::s_strides),
                index_t{},
                std::plus{},
                [](index_t const i, index_t const s) { return (i / Fanout) * s; }
            );
            auto out_patch_idx = typename patch_shape_t::multi_index_t{};
            do
            {
                const auto base = std::transform_reduce(
                                      std::cbegin(out_patch_idx),
                                      std::cend(out_patch_idx),
                                      std::cbegin(sized_strides),
                                      index_t{}
                                  ) /
                                  Fanout;
                to[out_patch_idx][idx] = offset + base;
            } while (out_patch_idx.increment());
        } while (idx.increment());
        return to;
    }();
    // -------------------------------------

    std::cout << "Patch index:\n" << patch_index_to << '\n';
}
