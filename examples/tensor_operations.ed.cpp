#include "containers/container_algorithms.hpp"
#include "containers/container_manipulations.hpp"
#include "containers/static_layout.hpp"
#include "containers/static_shape.hpp"
#include "containers/static_tensor.hpp"
#include "containers/static_vector.hpp"
#include "utility/random.hpp"
#include <cassert>
#include <cstdlib>
#include <iostream>

int main()
{
    using namespace amr::containers;
    constexpr auto N = 3;
    constexpr auto M = 2;
    using F          = int;
    using tensor5_t   = static_tensor<F, static_layout<static_shape<N, 3, 4, 2, 3>>>;
    using tensor2_t   = static_tensor<F, static_layout<static_shape<N, N, N>>>;
    using tensor3_t   = static_tensor<F, static_layout<static_shape<M, M>>>;
    std::cout << tensor5_t::elements() << '\n';
    for (int i = 0; i != tensor5_t::rank(); ++i)
    {
        std::cout << tensor5_t::size(i) << ", ";
    }
    std::cout << '\n';
    for (int i = 0; i != tensor5_t::rank(); ++i)
    {
        std::cout << "i: " << i << '\n';
        std::cout << "size: " << tensor5_t::size(i) << '\n';
        std::cout << "stride: " << tensor5_t::stride(i) << '\n';
    }
    tensor5_t t{};
    F        check{};
    std::iota(std::begin(t), std::end(t), check);
    amr::containers::manipulators::apply(
        t,
        [&check](auto const& a, auto... idxs)
        {
            const auto& e = a[idxs...];
            std::cout << e << '\n';
            assert((e == check++));
        }
    );

    auto idx = typename tensor5_t::multi_index_t{};
    do
    {
        std::cout << idx << " -> " << tensor5_t::linear_index(idx) << '\n';
    } while (idx.increment());
    std::cout << idx << '\n';

    std::cout << t << '\n';

    constexpr amr::containers::static_vector<float, 5> nodes{
        0.1f, 0.3f, 0.5f, 0.7f, 0.9f
    };
    constexpr auto reference_element =
        amr::containers::algorithms::tensor::cartesian_expansion<3>(nodes);
    std::cout << reference_element << '\n';

    for (auto i = typename decltype(reference_element)::multi_index_t{};;)
    {
        std::cout << i << " -> " << reference_element[i] << '\n';
        if (!i.increment())
        {
            break;
        }
    }

    tensor2_t t2{};
    tensor3_t t3{};
    std::iota(std::begin(t2), std::end(t2), 0);
    std::iota(std::begin(t3), std::end(t3), 0);
    std::cout << "Tensor product: \n";
    std::cout << "Rank 2 tensor\n" << t2 << '\n';
    std::cout << "Rank 3 tensor\n" << t3 << '\n';
    const auto tprod = amr::containers::algorithms::tensor::tensor_product(t2, t3);
    std::cout << tprod << '\n';

    return EXIT_SUCCESS;
}
