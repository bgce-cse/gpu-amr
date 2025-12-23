#include "containers/container_operations.hpp"
#include "containers/static_matrix.hpp"
#include "containers/static_tensor.hpp"
#include "containers/static_vector.hpp"
#include <array>
#include <iostream>
#include <vector>

int main()
{
    constexpr auto N = 12;
    using namespace amr::containers;
    using F        = double;
    using vec_t    = amr::containers::static_vector<F, N>;
    using mat_t    = amr::containers::static_matrix<F, 3, 4>;
    using tensor_t = amr::containers::
        static_tensor<float, static_layout<static_shape<std::array{ 2, 2, 3 }>>>;

    vec_t a{ 3.0, 9.0, 2.0, 6.0, 7.0, 9.0, 8.0, 0.0, 6.0, 10.0, 1.0, 2.0 };
    vec_t b{ 8.0, 9.0, 15.0, 92.0, 75.0, 093.0, 75.0, 3.0, 4.0, -5, 4.0, -1.0 };
    mat_t c{ 8.0, 7.0, 17.0, 89.0, 75.0, 093.0, 75.0, 3.0, 4.0, -5, -3.0, -3.0 };
    tensor_t d{ 10.f, 9.f, 15.f, 92.f, 75.f, 093.f, 75.f, 3.f, 4.f, -5, 2.f, 3.6f };

    std::cout << "a + b:\t" << a + b << '\n';
    std::cout << "(a + 0.5) * b:\t" << (a + 0.5) * b << '\n';
    std::cout << "2 * b + 4:\t" << 2 * b + 4 << '\n';
    std::cout << "2 * c + 5:\t" << 2 * c + 5 << '\n';
    std::cout << "2 * d + 5:\t" << 2 * d + 5 << '\n';
    std::cout << typeid(N).name() << N << '\n';
    return (int)(a + b)[5];
}
