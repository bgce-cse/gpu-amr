#include "containers/static_tensor.hpp"
#include <iostream>

int main()
{
    constexpr auto N = 6;
    constexpr auto M = 8;
    [[maybe_unused]]
    constexpr auto Fanout = 2;
    using tensor_t        = amr::containers::static_tensor<int, N, M>;

    tensor_t from{};
    std::ranges::iota(from, 0);
    std::cout << from << '\n';
    for (auto const& e : from)
        std::cout << e << " ";
    std::cout << '\n';

    tensor_t to[Fanout * Fanout]{};
    for (int i = 0; i != N; ++i)
    {
        for (int j = 0; j != M; ++j)
        {
            for (int ki = 0; ki != Fanout; ++ki)
            {
                for (int kj = 0; kj != Fanout; ++kj)
                {
                    to[ki * Fanout + kj][i, j] =
                        (i / Fanout) * M + (j / Fanout) + M / Fanout * (ki * N + kj);
                }
            }
        }
    }
    for (auto const& e : to)
        std::cout << e << '\n';
}
