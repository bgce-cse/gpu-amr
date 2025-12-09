#ifndef AMR_BASIS_LAGRANGE_HPP
#define AMR_BASIS_LAGRANGE_HPP

#include "containers/static_vector.hpp"
#include <stdexcept>
#include <type_traits>

namespace amr::basis
{

template <auto N>
using vector = amr::containers::static_vector<double, N>;

/**
 * @brief Lagrange interpolation utility for N nodes.
 * Fully constexpr, metaprogrammable.
 *
 * Provides function evaluation and derivative at compile-time.
 */
template <auto N>
struct Lagrange
{
    using size_type = std::conditional_t<
        std::is_same_v<decltype(N), unsigned int>,
        unsigned int,
        std::size_t>;

    /**
     * @brief Evaluate the i-th Lagrange polynomial at x.
     *
     * @param points Array of N nodal points
     * @param i Index of basis function
     * @param x Evaluation point
     * @return L_i(x)
     */
    static constexpr double evaluate(const vector<N>& points, size_type i, double x)
    {
        double numerator{ 1.0 };
        double denominator{ 1.0 };
        for (size_type k = 0; k < N; ++k)
        {
            if (k == i) continue;
            numerator *= (x - points[k]);
            denominator *= (points[i] - points[k]);
        }
        return numerator / denominator;
    }

    /**
     * @brief Compute the derivative of the i-th Lagrange polynomial at x.
     *
     * @param points Array of N nodal points
     * @param i Index of basis function
     * @param x Evaluation point
     * @return L'_i(x)
     */
    static constexpr double derivative(const vector<N>& points, size_type i, double x)
    {
        double result = 0.0;

        for (size_type j = 0; j < N; ++j)
        {
            if (j == i) continue;

            double term = 1.0 / (points[i] - points[j]);
            for (size_type k = 0; k < N; ++k)
            {
                if (k == i || k == j) continue;
                term *= (x - points[k]) / (points[i] - points[k]);
            }
            result += term;
        }
        return result;
    }
};

} // namespace amr::basis

#endif // AMR_BASIS_LAGRANGE_HPP
