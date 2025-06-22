#ifndef BASIS_HPP
#define BASIS_HPP

// #include "../containers/container_functions.hpp"
// #include "../containers/container_manipulations.hpp"
// #include "../containers/container_operations.hpp"
//#include "../containers/static_matrix.hpp"
#include "../containers/static_vector.hpp"
#include "../utility/constexpr_functions.hpp"
#include "globals.hpp"
// #include "../utility/random.hpp"
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>

namespace amr::basis
{



using coord3d = std::array<double, 3>;

template <std::size_t N>
constexpr double lagrange_1d(const std::array<double, N>& points, std::size_t i, double x) {
    if (i >= N) throw std::out_of_range("Index out of range");
    double numerator = 1.0f;
    double denominator = 1.0f;

    for (std::size_t k = 0; k < N; ++k) {
        if (k == i) continue;
        numerator *= (x - points[k]);
        denominator *= (points[i] - points[k]);
    }

    return numerator / denominator;
}

template <std::size_t N>
constexpr double lagrange_diff(const std::array<double, N>& points, std::size_t j, double x) {
    if (j >= N) throw std::out_of_range("Index out of range");
    double result = 0.0f;

    for (std::size_t i = 0; i < N; ++i) {
        if (i == j) continue;

        double term = 1.0f / (points[j] - points[i]);
        for (std::size_t k = 0; k < N; ++k) {
            if (k == i || k == j) continue;
            term *= (x - points[k]) / (points[j] - points[k]);
        }
        result += term;
    }

    return result;
}



template <unsigned int Order>
class GaussLegendre {
private:
    std::array<double, Order> _quadpoints_1d;
    std::array<double, Order> _quadweights_1d;

    double _start = -1.0f;
    double _end = 1.0f;

public:
    constexpr static unsigned int _order = Order;

    // Constructor with optional interval [start, end]
    GaussLegendre(double start = -1.0f, double end = 1.0f) : _start(start), _end(end) {
        compute_quadrature();
    }

    const std::array<double, Order>& points() const { return _quadpoints_1d; }
    const std::array<double, Order>& weights() const { return _quadweights_1d; }

private:
    // Derivative of Legendre polynomial P_n at x
    static double legendre_derivative(unsigned int n, double x) {
        // P_n'(x) = n/(1-x^2) * (x*P_n(x) - P_{n-1}(x))
        double Pn = std::legendre(n, x);
        double Pn_1 = std::legendre(n - 1, x);
        return n * (x * Pn - Pn_1) / (x * x - 1.0f);
    }

    void compute_quadrature() {
        constexpr double EPS = 1e-7f;
        unsigned int m = (Order + 1) / 2;

        for (unsigned int i = 0; i < m; ++i) {
            // Initial guess using Chebyshev nodes
            double z = std::cos(M_PI * (i + 0.75f) / (Order + 0.5f));
            double z1;

            // Newton-Raphson iteration to find root of P_n(x)
            do {
                z1 = z;
                double Pn = std::legendre(Order, z);
                double Pn_prime = legendre_derivative(Order, z);
                z = z1 - Pn / Pn_prime;
            } while (std::abs(z - z1) > EPS);

            // Compute weight
            double pp = legendre_derivative(Order, z);
            double w = 2.0f / ((1.0f - z * z) * pp * pp);

            // Store symmetric roots and weights
            _quadpoints_1d[i] = z;
            _quadpoints_1d[Order - 1 - i] = -z;
            _quadweights_1d[i] = w;
            _quadweights_1d[Order - 1 - i] = w;
        }

        // Map points and weights from [-1,1] to [start,end]
        double half_length = (_end - _start) / 2.0f;
        double midpoint = (_start + _end) / 2.0f;
        for (unsigned int i = 0; i < Order; ++i) {
            _quadpoints_1d[i] = midpoint + half_length * _quadpoints_1d[i];
            _quadweights_1d[i] *= half_length;
        }

        // sort points ascending 
        for (unsigned int i = 0; i < Order / 2; ++i) {
            if (_quadpoints_1d[i] > _quadpoints_1d[Order - 1 - i]) {
                std::swap(_quadpoints_1d[i], _quadpoints_1d[Order - 1 - i]);
                std::swap(_quadweights_1d[i], _quadweights_1d[Order - 1 - i]);
            }
        }
    }
};

template <unsigned int Dimensions, unsigned int Order>
class Basis{
private:
    
    static constexpr unsigned int _dimensions = Dimensions;
    GaussLegendre<Order> _gl;
public:
    
    Basis(double start = -1.0, double end = 1.0) : _gl(start, end) {}
    const GaussLegendre<Order>& getGL() const { return _gl; }
    const unsigned int dimensions() const { return _dimensions; }

    using coeff_matrix_t = amr::containers::static_vector<double, utility::cx_functions::pow(Order, Dimensions)>;

    auto evaluate_basis(const coeff_matrix_t& coeffs, const coord3d& v) const {
        std::array<double, Order> lx{}, ly{}, lz{};
        const auto& points = _gl.points();

        for (std::size_t i = 0; i < Order; ++i) {
            lx[i] = lagrange_1d(points, i, v[0]);
            ly[i] = lagrange_1d(points, i, v[1]);
            lz[i] = lagrange_1d(points, i, v[2]);
        }

        double result = 0.0;

        for (std::size_t k = 0; k < Order; ++k) {
            for (std::size_t j = 0; j < Order; ++j) {
                for (std::size_t i = 0; i < Order; ++i) {
                    const std::size_t index = i + Order * (j + Order * k);
                    result += coeffs[index] * lx[i] * ly[j] * lz[k];
                }
            }
        }

        return result;
    }

    std::vector<std::vector<double>> project_to_reference_basis(
    const std::function<std::vector<double>(double, double, double)>& fun,
    std::size_t ndofs) const{
    const auto& pts = _gl.points();
    std::size_t n = Order;

    std::vector<std::vector<double>> M(n * n * n, std::vector<double>(ndofs));

    for (std::size_t k = 0; k < n; ++k) {
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t i = 0; i < n; ++i) {
                double x = pts[i];
                double y = pts[j];
                double z = pts[k];

                std::size_t index = i + n * (j + n * k);

                M[index] = fun(x, y, z);

                assert(M[index].size() == ndofs);
            }
        }
    }

    return M;
    }

    std::vector<double> mass_matrix_diagonal(std::size_t dimensions) {
    const auto& w = _gl.weights();
    std::size_t n = w.size();

    std::vector<double> M_diag = {1.0};  

    for (std::size_t dim = 0; dim < dimensions; ++dim) {
        std::vector<double> temp;
        temp.reserve(M_diag.size() * n);

        for (double val : M_diag) {
            for (double weight : w) {
                temp.push_back(val * weight);
            }
        }

        M_diag = std::move(temp);
    }

    return M_diag;
    }
    //placeholder for derivative matrix comment to test the rest
    // std::array<matrix_t, 3> derivative_matrices_3d() const {
    //     auto D = derivative_matrix_1d();
    //     matrix_t D_mat(Order, std::vector<double>(Order));
    //     for (std::size_t i = 0; i < Order; ++i){
    //         for (std::size_t j = 0; j < Order; ++j){
    //             D_mat[i][j] = D[i][j];
    //         }
    //     }
    //     auto I = identity(Order);

    //     // Dx = D ⊗ I ⊗ I
    //     matrix_t Dx = kron(kron(D_mat, I), I);

    //     // Dy = I ⊗ D ⊗ I
    //     matrix_t Dy = kron(kron(I, D_mat), I);

    //     // Dz = I ⊗ I ⊗ D
    //     matrix_t Dz = kron(kron(I, I), D_mat);

    //     return {Dx, Dy, Dz};//needs to be hcat
    // }


    std::vector<std::array<double, 3>> get_face_quadpoints(amr::globals::Face face, const amr::globals::Globals& gm) {
    static_assert(Dimensions == 3, "get_face_quadpoints is only for 3D Basis");

    const auto& quadpoints = getGL().points();
    std::vector<std::array<double, 3>> result;
    result.reserve(Order * Order);

    int fixed_dim = gm.normal_idxs.at(face);
    int sign = gm.normal_signs.at(face);

    // Fixed value is 0.0 or 1.0 depending on sign
    double fixed_value = (sign == -1) ? 0.0 : 1.0;

    // Determine which two dimensions to vary
    std::array<int, 2> varying_dims;
    int idx = 0;
    for (int d = 0; d < 3; ++d) {
        if (d != fixed_dim) {
            varying_dims[idx++] = d;
        }
    }

    for (auto a : quadpoints) {
        for (auto b : quadpoints) {
            std::array<double, 3> point;
            point[fixed_dim] = fixed_value;
            point[varying_dims[0]] = a;
            point[varying_dims[1]] = b;
            result.push_back(point);
        }
    }

    return result;
}

    ~Basis() = default;
};

}// namespace amr::basis

#endif // BASIS_HPP