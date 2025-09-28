#include "containers/static_tensor.hpp"
#include <iostream>
#include "containers/static_layout.hpp"

int main()
{
    constexpr auto N = 6;
    constexpr auto M = 8;
    constexpr auto K = 4;
    [[maybe_unused]]
    constexpr auto Fanout = 2;
    using tensor_t        = amr::containers::static_tensor<int, N, M, K>;

    tensor_t from{};
    std::ranges::iota(from, 0);
    std::cout << from << '\n';
    for (auto const& e : from)
        std::cout << e << " ";
    std::cout << '\n';

    tensor_t to[Fanout * Fanout * Fanout]{};
    for (int i = 0; i != N; ++i)
    {
        for (int j = 0; j != M; ++j)
        {
            for (int k = 0; k != K; ++k)
            {
                for (int ki = 0; ki != Fanout; ++ki)
                {
                    for (int kj = 0; kj != Fanout; ++kj)
                    {
                        for (int kk = 0; kk != Fanout; ++kk)
                        {
                            // to[ki * Fanout + kj][i, j] =
                            //     (i / Fanout) * M + (j / Fanout) + (M - M / Fanout) *
                            //     (ki * N + kj);
                            //
                            // to[ki * Fanout + kj][i, j] =
                            //     (i / Fanout) * M + (j / Fanout) +
                            //     ki * M * N / Fanout + kj * M / Fanout;
                            to[(ki * Fanout + kj) * Fanout + kk][i, j, k] =
                                (i / Fanout) * M * K + (j / Fanout) * K + (k / Fanout) +
                                ki * M * N * K / Fanout + kj * M * K / Fanout +
                                kk * K / Fanout;
                        }
                    }
                }
            }
        }
    }
    for (auto const& e : to)
        std::cout << e << '\n';
}
