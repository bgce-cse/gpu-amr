#pragma once

#include "containers/container_algorithms.hpp"
#include "containers/static_tensor.hpp"
#include "containers/static_vector.hpp"
#include "generated_config.hpp"
#include "globals/global_config.hpp"
#include "globals/globals.hpp"
#include <cassert>

namespace amr::volume
{

/**
 * @brief Buffers for volume term evaluation
 *
 * Stores intermediate results during volume integral computation:
 * - scaled_fluxcoeff: flux coefficients scaled by inverse Jacobian and quadrature weights
 * - chi: diagonal scaling tensor combining volume and quadrature weights
 */
template <typename global_t, typename Policy>
struct BuffersVolume
{
    using scaled_flux_t = typename global_t::flux_tensor_t;
    using chi_t         = typename global_t::chi_tensor_t;

    scaled_flux_t scaled_fluxcoeff;
    chi_t         chi;

    BuffersVolume()
        : scaled_fluxcoeff{}
        , chi{}
    {
    }
};

/**
 * @brief Volume term evaluator for DG discretization
 *
 * Computes the contribution of flux divergence to the RHS via:
 * du/dt = -∇·F(u)
 *
 * The volume integral uses:
 * - Reference derivative matrix (in reference coordinates)
 * - Inverse Jacobian for coordinate transformation
 * - Quadrature weights for integration
 */
template <typename global_t, typename Policy>
struct VolumeEvaluator
{
    using global_config                     = global_t::GlobalConfig;
    static constexpr auto& ref_deriv_matrix = global_config::reference_derivative_matrix;
    static constexpr auto& quad_weights     = global_config::quad_weights_nd;

    /**
     * @brief Evaluate volume term contribution to RHS
     *
     * Implements: celldu += D^T * (χ ⊙ flux_coeff)
     * where χ = diagonal(A ⊗ quadweights), A = inv_jacobian' * volume
     *
     * @param buffers Pre-allocated buffers for intermediate results
     * @param flux_coeff Flux coefficients (computed from F(u))
     * @param inverse_jacobian Inverse Jacobian (assumed diagonal)
     * @param volume Volume of the cell
     * @param celldu Output: RHS update (+=)
     */
    template <typename FluxTensorType, typename JacobianType, typename DOFType>
    static void evaluate(
        BuffersVolume<global_t, Policy>& buffers,
        const FluxTensorType&            flux_coeff,
        const JacobianType&              inverse_jacobian,
        double                           volume,
        DOFType&                         celldu
    )
    {
        // Compute scaling: A = inv_jacobian' * volume
        // For diagonal case: diag(A) = diag(inv_jacobian) * volume
        auto A_diag = inverse_jacobian; // Element-wise product with volume
        for (auto& elem : A_diag)
            elem *= volume;

        // Build diagonal chi: kron(Diagonal(diag(A)), Diagonal(quadweights))
        // For diagonal tensors, this is element-wise multiplication of the kronecker
        // product chi[i,j] = A[i % dim_A] * quadweights[j % dim_Q]
        compute_chi_tensor(buffers.chi, A_diag, quad_weights);

        // Scale flux coefficients: scaled = chi ⊙ flux_coeff (element-wise)
        apply_diagonal_scaling(buffers.scaled_fluxcoeff, buffers.chi, flux_coeff);

        // Apply derivative matrix: celldu += D^T * scaled
        apply_derivative_matrix(celldu, ref_deriv_matrix, buffers.scaled_fluxcoeff);
    }

private:
    /**
     * @brief Compute chi tensor (diagonal scaling from Jacobian and quadrature)
     *
     * chi represents the Kronecker product of diagonal matrices:
     * chi = kron(Diagonal(A), Diagonal(quadweights))
     */
    template <typename ChiType, typename JacobianDiagType, typename QuadWeightsType>
    static void compute_chi_tensor(
        ChiType&                chi,
        const JacobianDiagType& jacobian_diag,
        const QuadWeightsType&  quad_weights
    )
    {
        // For Kronecker product of diagonal matrices:
        // (D_A ⊗ D_Q)[i + j*n_A, i + j*n_A] = A[i,i] * Q[j,j]
        std::size_t idx = 0;
        for (auto& elem : chi)
        {
            // Flatten 2D index into 1D
            auto i_a = idx % jacobian_diag.size();
            auto i_q = idx / jacobian_diag.size();
            elem     = jacobian_diag[i_a] * quad_weights[i_q];
            ++idx;
        }
    }

    /**
     * @brief Apply diagonal scaling: output = diag(chi) ⊙ input
     *
     * Element-wise multiplication with diagonal tensor
     */
    template <typename OutputType, typename ChiType, typename InputType>
    static void apply_diagonal_scaling(
        OutputType&      output,
        const ChiType&   chi,
        const InputType& input
    )
    {
        std::size_t idx = 0;
        for (auto& out_elem : output)
        {
            out_elem = chi[idx] * input[idx];
            ++idx;
        }
    }

    /**
     * @brief Apply derivative matrix: output += D^T * input
     *
     * Implements matrix-tensor contraction
     */
    template <typename OutputType, typename DerivMatrixType, typename InputType>
    static void apply_derivative_matrix(
        OutputType&            output,
        const DerivMatrixType& deriv_matrix,
        const InputType&       input
    )
    {
        // D^T * input where D is reference_derivative_matrix
        // This is a tensor contraction along the flux coefficient dimension
        auto result = amr::containers::algorithms::tensor::matrix_tensor_product(
            deriv_matrix, input
        );
        output = output + result;
    }
};

} // namespace amr::volume
