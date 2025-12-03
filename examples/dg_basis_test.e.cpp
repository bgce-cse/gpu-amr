#include "../include/dg_helpers/basis.hpp"
#include "../include/dg_helpers/globals.hpp"
#include "../include/dg_helpers/equations.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include <numeric>

// Include auto-generated configuration constants from config.yaml
#include "generated_config.hpp"

using namespace amr::Basis;
using namespace amr::config;

void test_vector_projection_3d() {
    std::cout << "Test 3D Vector Projection (f = [x+y+z, x*y*z])" << std::endl;
    std::cout << "============================================" << std::endl;

    // Use configuration constants from config.yaml
    constexpr unsigned int Order_c = Order;
    constexpr unsigned int Dim_c   = (Dim >= 3) ? 3 : 2;  // Force 3D for this test
    constexpr unsigned int DOFs_c  = 2;

    Basis<Order_c, Dim_c> basis(0.0, 1.0);
    using vec2 = amr::containers::static_vector<double, DOFs_c>;

    // Target vector-valued function
    auto fun = [](amr::containers::static_vector<double, Dim_c> pos) -> vec2 {
        return vec2{pos[0] + pos[1] + pos[2], pos[0] * pos[1] * pos[2]};
    };

    // Project onto reference basis
    auto coeffs = basis.project_to_reference_basis(fun);

    // Print coefficients for each quadrature node
    auto multi_idx = typename decltype(coeffs)::multi_index_t{};
    std::cout << "\nQuadrature nodes and coefficients:" << std::endl;
    while (true) {
        double x = basis.quadpoints()[multi_idx[0]];
        double y = basis.quadpoints()[multi_idx[1]];
        double z = basis.quadpoints()[multi_idx[2]];
        std::cout << std::fixed << std::setprecision(6)
                  << "Node (" << x << ", " << y << ", " << z << ") -> " << coeffs[multi_idx] << std::endl;
        if (!multi_idx.increment()) break;
    }

    // Random test point in [0,1]^3
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    amr::containers::static_vector<double, Dim_c> pos{dist(gen), dist(gen)};

    // Evaluate basis interpolation
    auto reconstructed = basis.evaluate_basis(coeffs, pos);
    auto exact = fun(pos);

    std::cout << "\nCheck at (x=" << pos[0] << ", y=" << pos[1] << ", z=" << pos[2] << "):" << std::endl;
    std::cout << "Reconstructed = " << reconstructed << std::endl;
    std::cout << "Exact         = " << exact << std::endl;
    std::cout << "Error         = "
              << amr::containers::static_vector<double, DOFs_c>{
                     std::abs(reconstructed[0] - exact[0]),
                     std::abs(reconstructed[1] - exact[1])}
              << std::endl;

    std::cout << "Config used: Order=" << Order_c << ", Dim=" << Dim_c << ", DOFs=" << DOFs_c << std::endl;
    std::cout << "============================================" << std::endl;
}

void test_global_face_kernels() {
    std::cout << "\nTest Global Face Kernels Dictionary" << std::endl;
    std::cout << "====================================" << std::endl;

    // Use configuration constants
    constexpr unsigned int Order_c = Order;
    constexpr unsigned int Dim_c   = (Dim >= 3) ? 3 : 2;  // Force 3D for this test

    // Create basis
    const Basis<Order_c, Dim_c> basis(0.0, 1.0);

    // Create face kernels dictionary
    amr::global::FaceKernels<Order_c, Dim_c> face_kernels(basis);

    // Test kernel at face coordinate 0
    std::cout << "\nFace Kernel at coordinate 0.0:" << std::endl;
    auto kernel_0 = face_kernels[0];
    for (unsigned int i = 0; i < Order_c; ++i) {
        std::cout << std::fixed << std::setprecision(8)
                  << "  phi[" << i << "] = " << kernel_0[i] << std::endl;
    }

    // Verify partition of unity at face coordinate 0
    double sum_0 = std::accumulate(kernel_0.begin(), kernel_0.end(), 0.0);
    std::cout << "Sum of phi at coord 0.0: " << sum_0 
              << " (should be 1.0)" << std::endl;

    // Test kernel at face coordinate 1
    std::cout << "\nFace Kernel at coordinate 1.0:" << std::endl;
    auto kernel_1 = face_kernels[1];
    for (unsigned int i = 0; i < Order_c; ++i) {
        std::cout << std::fixed << std::setprecision(8)
                  << "  phi[" << i << "] = " << kernel_1[i] << std::endl;
    }

    // Verify partition of unity at face coordinate 1
    double sum_1 = std::accumulate(kernel_1.begin(), kernel_1.end(), 0.0);
    std::cout << "Sum of phi at coord 1.0: " << sum_1 
              << " (should be 1.0)" << std::endl;

    // Test that kernels are accessible at compile time
    std::cout << "\nCompile-time accessibility check:" << std::endl;
    std::cout << "  First element of kernel[0]: " << face_kernels[0][0] << std::endl;
    std::cout << "  Last element of kernel[1]: " << face_kernels[1][Order_c-1] << std::endl;

    std::cout << "Config used: Order=" << Order_c << ", Dim=" << Dim_c << std::endl;
    std::cout << "====================================" << std::endl;
}

int main() {
    test_vector_projection_3d();
    test_global_face_kernels();
    return 0;
}
