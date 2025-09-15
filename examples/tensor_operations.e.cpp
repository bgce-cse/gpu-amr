#include "containers/static_tensor.hpp"
#include "utility/random.hpp"
#include <cassert>
#include <cstdlib>
#include <iostream>

int main()
{
    constexpr auto N = 5;
    using F          = int;
    using tensor_t   = amr::containers::static_tensor<F, N, 3, 4, 2, 3>;

    std::cout << tensor_t::flat_size() << '\n';
    for (int i = 0; i != tensor_t::rank(); ++i)
    {
        std::cout << tensor_t::size(i) << ", ";
    }
    std::cout << '\n';
    for (int i = 0; i != tensor_t::rank(); ++i)
    {
        std::cout << "i: " << i << '\n';
        std::cout << "size: " << tensor_t::size(i) << '\n';
        std::cout << "stride: " << tensor_t::stride(i) << '\n';
    }
    tensor_t t{};
    F        check{};
    std::iota(std::begin(t), std::end(t), check);
    for (int i = 0; i != N; ++i)
        for (int j = 0; j != 3; ++j)
            for (int k = 0; k != 4; ++k)
                for (int l = 0; l != 2; ++l)
                    for (int m = 0; m != 3; ++m)
                    {
                        std::cout << t[i, j, k, l, m] << '\n';
                        assert((t[i, j, k, l, m] == check++));
                    }

    auto idx = typename tensor_t::multi_index_t{};

    do
    {
        std::cout << idx << " -> " << tensor_t::linear_index(idx) << '\n';
    } while (idx.increment());
    std::cout << idx << '\n';

    std::cout << t << '\n';

    constexpr std::array<float, 5> nodes{ 0.1f, 0.3f, 0.5f, 0.7f, 0.9f };
    constexpr auto reference_element = amr::containers::cartesian_expansion<3>(nodes);
    std::cout << reference_element << '\n';

    return EXIT_SUCCESS;
}
