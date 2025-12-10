#include "containers/container_functions.hpp"
#include "containers/container_manipulations.hpp"
#include "containers/container_operations.hpp"
#include "containers/static_matrix.hpp"
#include "utility/random.hpp"
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

int main()
{
    using namespace amr::containers;
    constexpr auto N   = 5;
    using F            = double;
    using matrix_N_N_t = amr::containers::static_matrix<F, N, N>;
    using matrix_5_2_t = amr::containers::static_matrix<F, 5, 2>;

    using rng = typename utility::random::srandom;
    rng::seed<F>();

    matrix_N_N_t a;
    matrix_N_N_t b;
    matrix_5_2_t c;

    manipulators::fill(a, F{ 2 });
    manipulators::fill(b, []() { return rng::randfloat<F>(); });
    manipulators::fill(c, []() { return rng::randfloat<F>(); });

    std::cout << "a:\t" << a << '\n';
    std::cout << "b:\t" << b << '\n';
    std::cout << "c:\t" << c << '\n';
    // std::cout << "a * b:\t" << a * b << '\n';
    std::cout << "ab:\t" << amr::containers::matrix_mul(a, b) << '\n';
    std::cout << "L2 dist(a,b):\t"
              << std::sqrt(
                     amr::containers::distance<float>(
                         a,
                         b,
                         [](auto acc, auto v1, auto v2)
                         {
                             const auto d = v1 - v2;
                             return acc + static_cast<float>(d * d);
                         }
                     )
                 )
              << '\n';
    // std::cout << "a + b:\t" << a + b << '\n';
    // std::cout << "(a + 0.5) * b:\t" << (a + 0.5) * b << '\n';
    // std::cout << "2 * b + 4:\t" << 2 * b + 4 << '\n';
    // std::cout << "c + ca:\t" << c + ca << '\n';

    return EXIT_SUCCESS;
}
