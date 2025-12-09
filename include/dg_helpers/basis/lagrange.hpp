#ifndef AMR_BASIS_LAGRANGE_HPP
#define AMR_BASIS_LAGRANGE_HPP

#include "containers/static_vector.hpp"
#include <cstddef> // std::size_t
#include <type_traits>

namespace amr::basis
{

template <auto N>
using vector = amr::containers::static_vector<double, N>;

/**
 * @brief Lagrange interpolation utils (standard + barycentric).
 *
 * - Standard `evaluate` / `derivative` kept for compatibility.
 * - True barycentric `evaluate_barycentric` / `derivative_barycentric`.
 *
 * All functions are constexpr and safe for compile-time use.
 */
template <auto N>
struct Lagrange
{
    using size_type = std::conditional_t<
        std::is_same_v<decltype(N), unsigned int>,
        unsigned int,
        std::size_t>;

    static_assert(N >= 1, "Number of nodes N must be at least 1");

    /* -----------------------------
     * Standard Lagrange (classical)
     * ----------------------------- */

    /**
     * @brief Standard Lagrange basis L_i(x).
     */
    static constexpr double evaluate(const vector<N>& points, size_type i, double x)
    {
        double numerator   = 1.0;
        double denominator = 1.0;
        for (size_type k = 0; k < N; ++k)
        {
            if (k == i) continue;
            numerator *= (x - points[k]);
            denominator *= (points[i] - points[k]);
        }
        return numerator / denominator;
    }

    /**
     * @brief Standard derivative L'_i(x) (classical formulation).
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

    /**
     * @brief Compute barycentric weights w_i = 1 / Prod_{j != i} (x_i - x_j)
     *
     * Fully constexpr.
     */
    static constexpr vector<N> compute_barycentric_weights(const vector<N>& points)
    {
        vector<N> weights{};
        for (size_type i = 0; i < N; ++i)
        {
            double w = 1.0;
            for (size_type j = 0; j < N; ++j)
            {
                if (i == j) continue;
                w *= (points[i] - points[j]);
            }
            // w currently = Prod_{j!=i} (x_i - x_j)
            // store reciprocal to match definition w_i = 1 / Prod(...)
            weights[i] = 1.0 / w;
        }
        return weights;
    }

    /* --------------------------------
     * True barycentric evaluation
     * --------------------------------
     *
     * L_i(x) = (w_i / (x - x_i)) / S(x),  where S(x) = sum_k w_k / (x - x_k)
     *
     * Handle x == x_j exactly (return Kronecker delta).
     */

    static constexpr double evaluate_barycentric(
        const vector<N>& points,
        const vector<N>& weights,
        size_type        i,
        double           x
    )
    {
// If x exactly equals a node, return Kronecker delta
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
        for (size_type j = 0; j < N; ++j)
        {
            if (x == points[j]) return (i == j) ? 1.0 : 0.0;
        }
#pragma GCC diagnostic pop

        // Compute numerator and denominator
        double numerator = weights[i] / (x - points[i]);
        double denom     = 0.0;
        for (size_type j = 0; j < N; ++j)
            denom += weights[j] / (x - points[j]);

        return numerator / denom;
    }

    /* --------------------------------
     * True barycentric derivative
     * --------------------------------
     *
     * Use A(x) = w_i/(x-x_i), S(x) = sum_k w_k/(x-x_k)
     * L_i(x) = A(x) / S(x)
     * dA = -w_i/(x-x_i)^2
     * dS = - sum_k w_k/(x-x_k)^2
     *
     * L'_i = (dA * S - A * dS) / S^2
     *
     * Special cases:
     * - If x == x_j:
     *     * For i == j: L'_j(x_j) = sum_{k != j} 1 / (x_j - x_k)
     *     * For i != j: L'_i(x_j) = (w_i / w_j) * 1 / (x_j - x_i)
     */
    static constexpr double derivative_barycentric(
        const vector<N>& points,
        const vector<N>& weights,
        size_type        i,
        double           x
    )
    {
// node coincidence handling
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
        for (size_type j = 0; j < N; ++j)
        {
            if (x == points[j])
            {
                if (i == j)
                {
                    // L'_i(x_i) = sum_{k != i} 1/(x_i - x_k)
                    double sum = 0.0;
                    for (size_type k = 0; k < N; ++k)
                    {
                        if (k == i) continue;
                        sum += 1.0 / (points[i] - points[k]);
                    }
                    return sum;
                }
                else
                {
                    // L'_i(x_j) = (w_i / w_j) * 1/(x_j - x_i)
                    return (weights[i] / weights[j]) * (1.0 / (points[j] - points[i]));
                }
            }
        }
#pragma GCC diagnostic pop

        // General x != any node
        double A  = weights[i] / (x - points[i]);                      // A(x)
        double dA = -weights[i] / ((x - points[i]) * (x - points[i])); // dA
        double S  = 0.0;
        double dS = 0.0;
        for (size_type k = 0; k < N; ++k)
        {
            double term = weights[k] / (x - points[k]);
            S += term;
            dS += -weights[k] / ((x - points[k]) * (x - points[k])); // derivative of term
        }

        // (dA * S - A * dS) / S^2
        double num = dA * S - A * dS;
        double den = S * S;
        return num / den;
    }
};

} // namespace amr::basis

#endif // AMR_BASIS_LAGRANGE_HPP
