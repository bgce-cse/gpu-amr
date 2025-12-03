#include "containers/container_algorithms.hpp"
#include "containers/container_manipulations.hpp"
#include "containers/container_utils.hpp"
#include "containers/static_layout.hpp"
#include "containers/static_shape.hpp"
#include "containers/static_tensor.hpp"
#include "containers/static_vector.hpp"
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>

int main()
{
    using namespace amr::containers;

    // Helper for floating-point comparison
    auto approx_equal = [](double a, double b, double eps = 1e-10) -> bool
    {
        return std::abs(a - b) < eps;
    };
    constexpr auto N = 5;
    using F          = int;
    using tensor_t   = static_tensor<F, static_layout<static_shape<N, 3, 4, 2, 3>>>;
    std::cout << tensor_t::elements() << '\n';
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
    amr::containers::manipulators::apply(
        t,
        [&check](auto const& a, auto... idxs)
        {
            const auto& e = a[idxs...];
            std::cout << e << '\n';
            assert((e == check++));
        }
    );
    // amr::containers::manipulators::apply(
    //     t,
    //     [&check](auto const& a, auto... idxs)
    //     {
    //         const auto& e = a[idxs...];
    //         std::cout << e << '\n';
    //         assert((e == check++));
    //     }
    // );

    auto idx = typename tensor_t::multi_index_t{};
    do
    {
        std::cout << idx << " -> " << tensor_t::linear_index(idx) << '\n';
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

    // Test einsum_apply: Apply vector element-wise along different dimensions
    std::cout << "\n=== Testing einsum_apply ===\n";

    // Test case 1: 3D hypercube tensor [2,2,2] with vector applied along different
    // dimensions
    using tensor_3d_t = utils::types::tensor::hypercube_t<double, 2, 3>;
    tensor_3d_t tensor_3d{};
    // Fill tensor with sequential values
    double val = 1.0;
    for (auto it = std::begin(tensor_3d); it != std::end(tensor_3d); ++it)
    {
        *it = val++;
    }

    static_vector<double, 2> vec_dim0{ 2.0, 3.0 };
    auto result_dim0 = manipulators::einsum_apply<0>(tensor_3d, vec_dim0);
    std::cout << "Original 3D hypercube tensor shape [2,2,2]:\n" << tensor_3d << '\n';
    std::cout << "Vector applied along dim 0 [2.0, 3.0]:\n" << result_dim0 << '\n';

    // Verify: tensor[i,j,k] * vec[i] for all j,k
    auto verify_idx = typename tensor_3d_t::multi_index_t{};
    do
    {
        auto expected = tensor_3d[verify_idx] * vec_dim0[verify_idx[0]];
        assert(approx_equal(result_dim0[verify_idx], expected));
    } while (verify_idx.increment());
    std::cout << "✓ Dimension 0 application verified\n";

    // Test case 2: Vector applied along dimension 1
    static_vector<double, 2> vec_dim1{ 5.0, 6.0 };
    auto result_dim1 = manipulators::einsum_apply<1>(tensor_3d, vec_dim1);
    std::cout << "\nVector applied along dim 1 [5.0, 6.0]:\n" << result_dim1 << '\n';

    // Verify dimension 1 application
    verify_idx = typename tensor_3d_t::multi_index_t{};
    do
    {
        auto expected = tensor_3d[verify_idx] * vec_dim1[verify_idx[1]];
        assert(approx_equal(result_dim1[verify_idx], expected));
    } while (verify_idx.increment());
    std::cout << "✓ Dimension 1 application verified\n";

    // Test case 3: Vector applied along dimension 2
    static_vector<double, 2> vec_dim2{ 10.0, 20.0 };
    auto result_dim2 = manipulators::einsum_apply<2>(tensor_3d, vec_dim2);
    std::cout << "\nVector applied along dim 2 [10.0, 20.0]:\n" << result_dim2 << '\n';

    // Verify dimension 2 application
    verify_idx = typename tensor_3d_t::multi_index_t{};
    do
    {
        auto expected = tensor_3d[verify_idx] * vec_dim2[verify_idx[2]];
        assert(approx_equal(result_dim2[verify_idx], expected));
    } while (verify_idx.increment());
    std::cout << "✓ Dimension 2 application verified\n";

    // Test case 4: einsum_apply_op with custom operation
    // NOTE: einsum_apply_op has been replaced with more general einsum_apply
    // std::cout << "\n=== Testing einsum_apply_op with custom operation ===\n";
    // auto result_add =
    //     manipulators::einsum_apply_op<0>(tensor_3d, vec_dim0, std::plus<double>{});
    // std::cout << "Vector [2.0, 3.0, 4.0] added to tensor along dim 0:\n"
    //           << result_add << '\n';

    // Verify addition operation
    // verify_idx = typename tensor_3d_t::multi_index_t{};
    // do
    // {
    //     auto expected = tensor_3d[verify_idx] + vec_dim0[verify_idx[0]];
    //     assert(approx_equal(result_add[verify_idx], expected));
    // } while (verify_idx.increment());
    // std::cout << "✓ Addition operation verified\n";

    // Test with division
    // auto result_div = manipulators::einsum_apply_op<1>(
    //     tensor_3d, static_vector<double, 2>{ 2.0, 4.0 }, std::divides<double>{}
    // );
    // std::cout << "\nTensor divided by [2.0, 4.0] along dim 1:\n" << result_div << '\n';

    // Verify division operation
    // verify_idx = typename tensor_3d_t::multi_index_t{};
    // do
    // {
    // auto expected = tensor_3d[verify_idx] / (verify_idx[1] == 0 ? 2.0 : 4.0);
    // assert(approx_equal(result_div[verify_idx], expected));
    // }
    // while (verify_idx.increment());
    // std::cout << "✓ Division operation verified\n";

    // Test case 5: contract - dot product contraction
    std::cout << "\n=== Testing contract ===\n";

    // Contract along dimension 0 (vector size must match dimension size = 2)
    static_vector<double, 2> contract_vec_dim0{ 1.0, 2.0 };
    auto result_contract_dim0 = manipulators::contract<0>(tensor_3d, contract_vec_dim0);
    std::cout << "Contract tensor along dim 0 with [1.0, 2.0]:\n"
              << result_contract_dim0 << '\n';
    std::cout << "✓ Dimension 0 contraction completed\n";

    // Contract along dimension 1
    static_vector<double, 2> contract_vec_dim1{ 1.0, 2.0 };
    auto result_contract_dim1 = manipulators::contract<1>(tensor_3d, contract_vec_dim1);
    std::cout << "\nContract tensor along dim 1 with [1.0, 2.0]:\n"
              << result_contract_dim1 << '\n';
    std::cout << "✓ Dimension 1 contraction completed\n";

    // Contract along dimension 2
    static_vector<double, 2> contract_vec_dim2{ 1.0, 2.0 };
    auto result_contract_dim2 = manipulators::contract<2>(tensor_3d, contract_vec_dim2);
    std::cout << "\nContract tensor along dim 2 with [1.0, 2.0]:\n"
              << result_contract_dim2 << '\n';
    std::cout << "✓ Dimension 2 contraction completed\n";

    std::cout << "\n✓ All einsum tests passed!\n";

    return EXIT_SUCCESS;
}
