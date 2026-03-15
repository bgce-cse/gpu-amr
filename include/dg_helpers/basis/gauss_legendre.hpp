#ifndef AMR_BASIS_GAUSS_LEGENDRE_HPP
#define AMR_BASIS_GAUSS_LEGENDRE_HPP

#include "containers/container_operations.hpp"
#include "containers/static_vector.hpp"
#include "dg_helpers/globals/quadrature.hpp"

namespace amr::basis
{

template <auto N>
using vector = amr::containers::static_vector<double, N>;

/**
 * @brief Precomputed 1D Gauss-Legendre quadrature on interval [start, end].
 * Fully compile-time available.
 */
template <auto Order, double Start = 0.0, double End = 1.0>
struct GaussLegendre
{
    static const constexpr vector<Order> points = []()
    {
        constexpr auto& precomputed = amr::global::QuadData<Order>::points;
        const double    half_length = (End - Start) / 2.0;
        const double    midpoint    = (Start + End) / 2.0;
        vector<Order>   result{};
        for (unsigned int i = 0; i < Order; ++i)
            result[i] = midpoint + half_length * precomputed[i];
        return result;
    }();

    static const constexpr vector<Order> weights = []()
    {
        constexpr auto& precomputed = amr::global::QuadData<Order>::weights;
        const double    half_length = (End - Start) / 2.0;
        vector<Order>   result{};
        for (unsigned int i = 0; i < Order; ++i)
            result[i] = precomputed[i] * half_length;
        return result;
    }();
};

} // namespace amr::basis

#endif // AMR_BASIS_GAUSS_LEGENDRE_HPP
