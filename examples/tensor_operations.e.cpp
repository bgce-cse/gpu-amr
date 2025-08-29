#include "containers/static_tensor.hpp"
#include "utility/random.hpp"
#include <cstdlib>
#include <iostream>

int main()
{
    constexpr auto N = 5;
    using F          = double;
    using tensor_t   = amr::containers::static_tensor<F, N, N, 4, N>;

    std::cout << tensor_t::flat_size() << '\n';
    for (int i = 0; i != tensor_t::s_dims; ++i)
    {
        std::cout << tensor_t::size(i) << '\n';
    }

    return EXIT_SUCCESS;
}
