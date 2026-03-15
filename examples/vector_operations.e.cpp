#include "containers/container_operations.hpp"
#include "containers/static_vector.hpp"
#include <array>
#include <iostream>
#include <vector>

int main()
{
    constexpr std::size_t N = 10;
    using F                 = double;
    using vec_t             = amr::containers::static_vector<F, N>;

    std::vector<float> vf{ 3.0f, .04f, 5.0f, 6.0f, 5.0f, 3.0f, 7.0f, 8.0f, .02f, -4.f };
    std::array<int, N> ai{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    F                  ca[N]{ 3.0, 4.0, 3.0, 3.0, 3.0, 0.0, 1.0, 2.0, 3.0, 4.0 };
    vec_t              a{ 3.0, 9.0, 2.0, 6.0, 7.0, 9.0, 8.0, 0.0, 6.0, 10.0 };
    vec_t              b{ 8.0, 9.0, 15.0, 92.0, 75.0, 093.0, 75.0, 3.0, 4.0, -5 };

    std::cout << "a + b:\t" << a + b << '\n';
    std::cout << "(a + 0.5) * b:\t" << (a + 0.5) * b << '\n';
    std::cout << "a + ai:\t" << a + ai << '\n';
    std::cout << "2 * b + 4:\t" << 2 * b + 4 << '\n';
    std::cout << "vf + a:\t" << vf + a << '\n';
    std::cout << "a + ca:\t" << a + ca << '\n';
    std::cout << typeid(N).name() << '\n';
    return (int)(a + b)[5];
}
