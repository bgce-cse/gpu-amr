#ifndef BASIS_HPP
#define BASIS_HPP
#include <torch/torch.h>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>
#include <cassert>
#include <functional>
namespace amr::Basis{

    

// Computes the Legendre polynomial P_n(x) using recurrence
// n: degree >= 0
// x: point in [-1, 1]
    inline double legendre(int n, double x) {
        assert(n >= 0 && "Legendre polynomial degree must be >= 0");

        if (n == 0) return 1.0;
        if (n == 1) return x;

        double Pnm2 = 1.0;       // P_0(x)
        double Pnm1 = x;         // P_1(x)
        double Pn;

        for (int k = 2; k <= n; ++k) {
            Pn = ((2.0 * k - 1.0) * x * Pnm1 - (k - 1.0) * Pnm2) / k;
            Pnm2 = Pnm1;
            Pnm1 = Pn;
        }
        return Pn;
    }
    /**
     * Evaluate the Lagrange interpolation polynomial defined over nodal `points` with index `i`
     * at point `x`.
     * 
     * @param points Vector of nodal points
     * @param i Index of the Lagrange basis function (0-based)
     * @param x Point at which to evaluate the polynomial
     * @return Value of the i-th Lagrange basis polynomial at x
     */
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
            double Pn = amr::Basis::legendre(n, x);
            double Pn_1 = amr::Basis::legendre(n - 1, x);
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
                    double Pn = amr::Basis::legendre(Order, z);
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
    template <unsigned int Order, unsigned int Dim>
    class Basis {
    private:
        std::array<std::array<double, Order>, Dim> _quadpoints;   // quadrature points per dimension
        std::array<std::array<double, Order>, Dim> _quadweights;
        int _dimensions;
        int _order;
        int _ndofs;

        // Store all multi-indices of size Order^Dim
        std::vector<std::array<unsigned int, Dim>> _spatial_indices;
        std::vector<std::array<unsigned int, Dim+1>> _dof_indices;

    public:
        Basis(double start = 0, double end = 1.0, int ndofs = 1.0)
            : _dimensions(Dim), _order(Order), _ndofs(ndofs)
        {
            GaussLegendre<Order> quad_1d(start, end);

            for (unsigned int d = 0; d < Dim; ++d) {
            _quadpoints[d] = quad_1d.points();
            _quadweights[d] = quad_1d.weights();
        }

            // Precompute all multi-indices for tensor product of size Order^Dim
            unsigned int total_points = static_cast<unsigned int>(std::pow(Order, Dim));
            _spatial_indices.reserve(total_points);

            for (unsigned int flat_idx = 0; flat_idx < total_points; ++flat_idx) {
                _spatial_indices.push_back(compute_spatial_index(flat_idx));
            }
            for (unsigned int flat_idx = 0; flat_idx < total_points * _ndofs; ++flat_idx) {
                _dof_indices.push_back(compute_dof_index(flat_idx));
            }
        }

        const auto& quadpoints() const { return _quadpoints; }
        const auto& quadweights() const { return _quadweights; }
        int order() const { return _order; }
        int dimensions() const { return _dimensions; }

        // Compute multi-index for given flat index (fastest dimension = 0)
        std::array<unsigned int, Dim> compute_spatial_index(unsigned int flat_index) {
            std::array<unsigned int, Dim> multi_index;
            for (int d = Dim - 1; d >= 0; --d) {
                multi_index[d] = flat_index % _order;
                flat_index /= _order;
            }
            return multi_index;
        }
        std::array<unsigned int, Dim + 1> compute_dof_index(unsigned int flat_index) {
            std::array<unsigned int, Dim + 1> multi_index;

            // The fastest running dimension: ndofs
            multi_index[Dim] = flat_index % _ndofs;
            flat_index /= _ndofs;

            // The remaining Dim spatial indices (slower running)
            for (int d = Dim - 1; d >= 0; --d) {
                multi_index[d] = flat_index % _order;
                flat_index /= _order;
            }

            return multi_index;
        }

        // Accessor for the precomputed multi-indices
        const std::vector<std::array<unsigned int, Dim>>& multi_indices() const {
            return _spatial_indices;
        }

        double evaluate_basis(torch::Tensor & coeffs, std::array<double,Dim> x)const {
            int64_t size = coeffs.numel();
            double * data = coeffs.data_ptr<double>();
            double sum = 0;
            for (int64_t i = 0; i < size; i++) {
                double prod = 1.0;
                for(int j = 0; j< _dimensions; ++j){
                    prod *= lagrange_1d(_quadpoints[j], _spatial_indices[i][j],x[j]);
                }
                sum += data[i]*prod;
            }
            return sum;
        }
        /***
        Project the result of the function `fun` to coefficients
        of the basis built of a tensor-product of `basis`.
        The function returns a vector with size `ndofs`.
        The corresponding coefficients are returned.
         */
        torch::Tensor project_to_reference_basis(std::function<std::vector<double>(const std::vector<double>&)> fun){
            auto eval = torch::zeros([&]() {
                            std::vector<int64_t> tensor_initializer;
                            for (int i = 0; i < _dimensions; ++i) {
                            tensor_initializer.push_back(_order);
                            }
                            tensor_initializer.push_back(_ndofs);
                            return tensor_initializer;
                        }(),
                        torch::kDouble
                        );
            int64_t size = eval.numel();
            double * data = eval.template data_ptr<double>();
            
            for(int64_t i = 0; i < size; i++){
                std::vector<double> position;
                for(int j = 0;j< _dimensions; j++){
                    position.push_back(_quadpoints.at(j).at(_dof_indices.at(i).at(j)));
                }
                std::cout << "Position: ";
                for (const auto& pos : position) {
                    std::cout << pos << " ";
                }
                std::cout << std::endl;
                data[i] = fun(position).at(_dof_indices.at(i).back());
            }
            
            return eval;
        }
    };


}
#endif