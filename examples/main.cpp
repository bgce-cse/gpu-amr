#include <torch/torch.h>
#include <iostream>
#include "../include/dg_helpers/basis.hpp"

#include <iostream>
#include <cmath>
#include <array>
#include <vector>
#include <iomanip>
#include <cassert>
#include <functional>

// Include your header file here
// #include "lagrange_basis.h"

// For testing purposes, assuming the code is in namespace amr::Basis
using namespace amr::Basis;

// Test utility functions
const double EPSILON = 1e-10;

bool approx_equal(double a, double b, double eps = EPSILON) {
    return std::abs(a - b) < eps;
}

void print_test_result(const std::string& test_name, bool passed) {
    std::cout << "[" << (passed ? "PASS" : "FAIL") << "] " << test_name << std::endl;
}

// Test 1: Basic Lagrange interpolation properties
void test_lagrange_basic_properties() {
    std::cout << "\n=== Testing Basic Lagrange Properties ===" << std::endl;
    
    // Test with 3 points: [-1, 0, 1]
    std::array<double, 3> points = {-1.0, 0.0, 1.0};
    
    // Test Kronecker delta property: L_i(x_j) = δ_ij
    bool kronecker_test = true;
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            double actual = lagrange_1d(points, i, points[j]);
            if (!approx_equal(actual, expected)) {
                kronecker_test = false;
                std::cout << "  Failed: L_" << i << "(" << points[j] << ") = " 
                         << actual << ", expected " << expected << std::endl;
            }
        }
    }
    print_test_result("Kronecker delta property", kronecker_test);
    
    // Test partition of unity: Σ L_i(x) = 1
    bool unity_test = true;
    std::vector<double> test_points = {-0.5, 0.3, 0.7};
    for (double x : test_points) {
        double sum = 0.0;
        for (std::size_t i = 0; i < 3; ++i) {
            sum += lagrange_1d(points, i, x);
        }
        if (!approx_equal(sum, 1.0)) {
            unity_test = false;
            std::cout << "  Failed: Σ L_i(" << x << ") = " << sum << ", expected 1.0" << std::endl;
        }
    }
    print_test_result("Partition of unity", unity_test);
}

// Add this test function after your existing test functions





// Test 2: Lagrange derivative properties
void test_lagrange_derivatives() {
    std::cout << "\n=== Testing Lagrange Derivatives ===" << std::endl;
    
    std::array<double, 3> points = {-1.0, 0.0, 1.0};
    
    // Test that derivative of partition of unity is zero: Σ L_i'(x) = 0
    bool derivative_unity_test = true;
    std::vector<double> test_points = {-0.5, 0.3, 0.7};
    for (double x : test_points) {
        double sum = 0.0;
        for (std::size_t i = 0; i < 3; ++i) {
            sum += lagrange_diff(points, i, x);
        }
        if (!approx_equal(sum, 0.0, 1e-8)) {
            derivative_unity_test = false;
            std::cout << "  Failed: Σ L_i'(" << x << ") = " << sum << ", expected 0.0" << std::endl;
        }
    }
    print_test_result("Derivative of partition of unity", derivative_unity_test);
    
    // Test specific derivative values for known case
    // For L_1(x) = (x+1)(x-1)/(-1*1) = -(x^2-1) = -x^2+1 on [-1,0,1]
    // L_1'(x) = -2x, so L_1'(0.5) = -1.0
    double expected_deriv = -1.0;
    double actual_deriv = lagrange_diff(points, 1, 0.5);
    bool specific_deriv_test = approx_equal(actual_deriv, expected_deriv, 1e-8);
    if (!specific_deriv_test) {
        std::cout << "  Failed: L_1'(0.5) = " << actual_deriv << ", expected " << expected_deriv << std::endl;
    }
    print_test_result("Specific derivative value", specific_deriv_test);
}

// Test 3: Gauss-Legendre quadrature accuracy
void test_gauss_legendre_accuracy() {
    std::cout << "\n=== Testing Gauss-Legendre Quadrature Accuracy ===" << std::endl;
    
    // Test integration of polynomials up to degree 2n-1
    // 2-point Gauss-Legendre should integrate polynomials up to degree 3 exactly
    GaussLegendre<2> quad2(-1.0, 1.0);
    auto points2 = quad2.points();
    auto weights2 = quad2.weights();
    
    // Test f(x) = 1 (degree 0), integral = 2
    double integral_const = 0.0;
    for (std::size_t i = 0; i < 2; ++i) {
        integral_const += weights2[i] * 1.0;
    }
    bool const_test = approx_equal(integral_const, 2.0);
    print_test_result("Constant function integration", const_test);
    
    // Test f(x) = x^2 (degree 2), integral = 2/3
    double integral_x2 = 0.0;
    for (std::size_t i = 0; i < 2; ++i) {
        integral_x2 += weights2[i] * (points2[i] * points2[i]);
    }
    bool x2_test = approx_equal(integral_x2, 2.0/3.0);
    print_test_result("x^2 integration", x2_test);
    
    // Test f(x) = x^3 (degree 3), integral = 0 (odd function)
    double integral_x3 = 0.0;
    for (std::size_t i = 0; i < 2; ++i) {
        integral_x3 += weights2[i] * (points2[i] * points2[i] * points2[i]);
    }
    bool x3_test = approx_equal(integral_x3, 0.0);
    print_test_result("x^3 integration", x3_test);
    
    // Test with higher order
    GaussLegendre<3> quad3(-1.0, 1.0);
    auto points3 = quad3.points();
    auto weights3 = quad3.weights();
    
    // Test f(x) = x^4 (degree 4), integral = 2/5
    double integral_x4 = 0.0;
    for (std::size_t i = 0; i < 3; ++i) {
        double x4 = points3[i] * points3[i] * points3[i] * points3[i];
        integral_x4 += weights3[i] * x4;
    }
    bool x4_test = approx_equal(integral_x4, 2.0/5.0);
    print_test_result("x^4 integration with 3-point rule", x4_test);
}

// Test 4: Interval transformation
void test_interval_transformation() {
    std::cout << "\n=== Testing Interval Transformation ===" << std::endl;
    
    // Test transformation from [-1,1] to [0,1]
    GaussLegendre<2> quad_01(0.0, 1.0);
    auto points_01 = quad_01.points();
    auto weights_01 = quad_01.weights();
    
    // Integrate f(x) = 1 over [0,1], should give 1
    double integral_01 = 0.0;
    for (std::size_t i = 0; i < 2; ++i) {
        integral_01 += weights_01[i] * 1.0;
    }
    bool interval_test = approx_equal(integral_01, 1.0);
    print_test_result("Integration over [0,1]", interval_test);
    
    // Test that points are in [0,1]
    bool points_in_range = true;
    for (std::size_t i = 0; i < 2; ++i) {
        if (points_01[i] < 0.0 || points_01[i] > 1.0) {
            points_in_range = false;
            break;
        }
    }
    print_test_result("Points in [0,1] range", points_in_range);
}

// Test 5: Edge cases and error handling
void test_edge_cases() {
    std::cout << "\n=== Testing Edge Cases ===" << std::endl;
    
    std::array<double, 3> points = {-1.0, 0.0, 1.0};
    
    // Test out of range index (should throw)
    bool exception_test = false;
    try {
        lagrange_1d(points, 3, 0.5);  // Invalid index
    } catch (const std::out_of_range&) {
        exception_test = true;
    }
    print_test_result("Out of range index exception", exception_test);
    
    // Test single point case
    std::array<double, 1> single_point = {0.5};
    double single_result = lagrange_1d(single_point, 0, 0.3);
    bool single_test = approx_equal(single_result, 1.0);
    print_test_result("Single point Lagrange", single_test);
    
    // Test GaussLegendre with order 1
    GaussLegendre<1> quad1(-1.0, 1.0);
    auto points1 = quad1.points();
    auto weights1 = quad1.weights();
    
    bool order1_test = approx_equal(points1[0], 0.0) && approx_equal(weights1[0], 2.0);
    print_test_result("Order 1 Gauss-Legendre", order1_test);
}

// Test 6: Numerical stability
void test_numerical_stability() {
    std::cout << "\n=== Testing Numerical Stability ===" << std::endl;
    
    // Test with closely spaced points
    std::array<double, 3> close_points = {0.0, 0.001, 0.002};
    
    // Should still satisfy partition of unity
    bool stability_test = true;
    double x = 0.0015;
    double sum = 0.0;
    for (std::size_t i = 0; i < 3; ++i) {
        sum += lagrange_1d(close_points, i, x);
    }
    if (!approx_equal(sum, 1.0, 1e-6)) {
        stability_test = false;
    }
    print_test_result("Partition of unity with close points", stability_test);
}


// Test 7: evaluate_basis function
void test_evaluate_basis() {
    std::cout << "\n=== Testing evaluate_basis Function ===" << std::endl;
    
    constexpr unsigned int Order = 3;
    constexpr unsigned int Dim = 2;
    
    // Create basis
    Basis<Order, Dim> basis(-1.0, 1.0, 4);
    
    // Create a simple coefficient tensor (3x3 for 2D)
auto coeffs = torch::ones({Order, Order}, torch::kFloat64);

    
    // Test point at origin
    std::array<double, Dim> x = {0.0, 0.0};
    
    // Evaluate basis
    double result = basis.evaluate_basis(coeffs, x);
    
    // With all coefficients = 1, and partition of unity property,
    // the result should be 1.0
    bool basic_test = approx_equal(result, 1.0, 1e-8);
    print_test_result("Basic evaluate_basis with ones", basic_test);
    
    // Test with zero coefficients
    auto zero_coeffs = torch::zeros({Order, Order}, torch::kFloat64);


    double zero_result = basis.evaluate_basis(zero_coeffs, x);
    bool zero_test = approx_equal(zero_result, 0.0, 1e-10);
    print_test_result("evaluate_basis with zeros", zero_test);
    
    // Test with single coefficient
    auto single_coeffs = torch::zeros({Order, Order}, torch::kFloat64);
    single_coeffs[1][1] = 2.0;  // Middle coefficient
    double single_result = basis.evaluate_basis(single_coeffs, x);
    
    // At origin, middle basis function should have value 1.0
    bool single_test = approx_equal(single_result, 2.0, 1e-8);
    print_test_result("evaluate_basis with single coefficient", single_test);
    
    std::cout << "  Result at origin with coeff[1][1]=2.0: " << single_result << std::endl;
}

void test_project_to_reference_basis_2d() {
    std::cout << "\n=== Testing project_to_reference_basis with 2D function ===" << std::endl;

    constexpr unsigned int Order = 5;
    constexpr unsigned int Dim = 3;
    constexpr unsigned int ndofs = 1;

    amr::Basis::Basis<Order, Dim> basis(0, 1.0,ndofs);

    auto test_fun = [](const std::vector<double>& x) -> std::vector<double> {
        std::vector<double> out{x[0]+ x[1]+x[2]}; // Simple function: f(x,y,z) = x + y + z
        assert(out.size()== ndofs);
        return out;
    };

    torch::Tensor eval = basis.project_to_reference_basis(test_fun);

    std::cout << eval << std::endl;
    print_test_result("project_to_reference_basis 2D function test", eval.numel() > 0);
}



int main() {
    std::cout << "Running Lagrange Interpolation and Gauss-Legendre Tests" << std::endl;
    std::cout << "=======================================================" << std::endl;
    
    test_lagrange_basic_properties();
    test_lagrange_derivatives();
    test_gauss_legendre_accuracy();
    test_interval_transformation();
    test_edge_cases();
    test_numerical_stability();
    test_evaluate_basis();
    test_project_to_reference_basis_2d();
    
    std::cout << "\n=======================================================" << std::endl;
    std::cout << "All tests completed!" << std::endl;
    constexpr unsigned int Order = 4;
    constexpr unsigned int Dim = 3;

    // Construct Basis
    Basis<Order, Dim> basis(-1.0, 1.0, 4);

    // 2️⃣ Test multi_indices generation
    auto multi_indices = basis.multi_indices();
    assert(multi_indices.size() == static_cast<size_t>(std::pow(Order, Dim)));

    std::cout << "Multi-indices:\n";
    for (const auto& idx : multi_indices) {
        for (auto v : idx) {
            std::cout << v << " ";
            assert(v < Order);
        }
        std::cout << "\n";
    }

    // 3️⃣ Test tensor shape matching
    auto coeffs = torch::rand({Order, Order,Order});
    std::cout << "Coefficients Tensor:\n" << coeffs << "\n";

    for (size_t i = 0; i < multi_indices.size(); ++i) {
        const auto& idx = multi_indices[i];
        // Convert std::array<unsigned int, Dim> to torch::indexing::TensorIndex
        std::vector<torch::indexing::TensorIndex> torch_idx;
        for (auto v : idx) {
            torch_idx.push_back(static_cast<int64_t>(v));
        }
        auto val = coeffs.index(torch_idx).item<double>();
        std::cout << "Tensor[" << idx[0] << "," << idx[1] << "] = " << val << "\n";
    }

    
    return 0;
}
