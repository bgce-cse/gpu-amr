#ifndef AMR_BASIS_POLYNOMIAL_HPP
#define AMR_BASIS_POLYNOMIAL_HPP

#include "containers/static_vector.hpp"
#include "gauss_legendre.hpp"
#include "lagrange.hpp"

namespace amr::basis
{

template <auto N>
using vector = amr::containers::static_vector<double, N>;

/**
 * @brief Tensor-product Lagrange basis with Gauss-Legendre quadrature.
 * Fully compile-time available.
 */
template <
    std::integral auto Order,
    std::integral auto Dim,
    double             Start = 0.0,
    double             End   = 1.0>
struct Basis
{
    using GaussLegendre_t = GaussLegendre<Order, Start, End>;
    using Lagrange_t      = Lagrange<Order>;
    using vector_t        = vector<Dim>;

    static constexpr const auto& quadpoints  = GaussLegendre_t::points;
    static constexpr const auto& quadweights = GaussLegendre_t::weights;

    // Evaluate tensor-product Lagrange basis at a given position
    template <typename CoeffTensor>
    [[nodiscard]]
    static constexpr auto
        evaluate_basis(const CoeffTensor& coeffs, const vector_t& position)
    {
        static_assert(
            CoeffTensor::rank() == Dim,
            "Coefficient tensor rank must match spatial dimension"
        );

        using value_t       = typename CoeffTensor::value_type;
        using multi_index_t = typename CoeffTensor::multi_index_t;

        value_t       sum{};
        multi_index_t idx{};

        do
        {
            double prod = 1.0;
            for (std::size_t j = 0; j < Dim; ++j)
                prod *= Lagrange_t::evaluate(quadpoints, idx[j], position[j]);
            sum = sum + coeffs[idx] * prod;
        } while (idx.increment());

        return sum;
    }

    // Project a function onto the reference tensor-product Lagrange basis
    template <typename Func>
    [[nodiscard]]
    static auto project_to_reference_basis(Func&& fun)
    {
        using return_t = decltype(fun(vector_t{}));
        using tensor_t = typename amr::containers::utils::types::tensor::
            hypercube_t<return_t, Order, Dim>;
        using multi_index_t = typename tensor_t::multi_index_t;

        tensor_t      coeffs{};
        multi_index_t idx{};

        do
        {
            vector_t position{};
            for (unsigned int d = 0; d < Dim; ++d)
                position[d] = quadpoints[idx[d]];
            coeffs[idx] = fun(position);

        } while (idx.increment());

        return coeffs;
    }

    // Create 1D face kernel vector
    [[nodiscard]]
    static constexpr auto create_face_kernel(double face_coord)
    {
        vector<Order> phi{};
        for (unsigned int i = 0; i < Order; ++i)
            phi[i] = Lagrange_t::evaluate(GaussLegendre_t::points, i, face_coord);
        return phi;
    }
};

} // namespace amr::basis

#endif // AMR_BASIS_POLYNOMIAL_HPP
