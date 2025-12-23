#include "containers/container_algorithms.hpp"
#include "containers/static_layout.hpp"
#include "containers/static_shape.hpp"
#include "containers/static_tensor.hpp"
#include "containers/static_vector.hpp"
#include <array>
#include <iostream>
#include <numeric>

int main()
{
    using F = int;
    using namespace amr::containers;
    using tensor_a_t =
        static_tensor<F, static_layout<static_shape<std::array{ 2, 2, 3 }>>>;
    using tensor_b_t = static_tensor<F, static_layout<static_shape<std::array{ 2, 2 }>>>;

    tensor_a_t a{};
    tensor_b_t b{};

    static_assert(std::ranges::range<tensor_a_t>);
    static_assert(std::ranges::range<tensor_b_t>);

    std::ranges::iota(a, F{});
    std::ranges::iota(b, F{});

    std::cout << typeid(tensor_a_t).name() << '\n' << a << '\n';
    std::cout << typeid(tensor_b_t).name() << '\n' << b << '\n';

    constexpr auto contraction_index_set1 =
        utils::types::tensor::contraction_index_set<int, 1>(std::pair{ 0, 0 });

    const auto c1 = algorithms::tensor::contraction<contraction_index_set1>(a, b);
    std::cout << c1 << '\n';

    return EXIT_SUCCESS;
}
