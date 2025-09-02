#include "containers/static_tensor.hpp"
#include "utility/random.hpp"
#include <cstdlib>
#include <iostream>
#include <cassert>

int main()
{
    constexpr auto N = 5;
    using F          = int;
    using tensor_t   = amr::containers::static_tensor<F, N, 3, 4, 2>;

    std::cout << tensor_t::flat_size() << '\n';
    for (int i = 0; i != tensor_t::rank(); ++i)
    {
        std::cout << "i: " << i << '\n';
        std::cout << "size: " << tensor_t::size(i) << '\n';
        std::cout << "stride: " << tensor_t::stride(i) << '\n';
    }
    tensor_t t{};
    std::iota(std::begin(t), std::end(t), F{});
    F check{};
    for (int i = 0; i != N; ++i)
        for (int j = 0; j != 3; ++j)
            for (int k = 0; k != 4; ++k)
                for (int l = 0; l != 2; ++l)
                    assert((t[i,j,k,l] == check++));

    return EXIT_SUCCESS;
}
