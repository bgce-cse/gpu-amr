#pragma once

#include "containers/container_algorithms.hpp"
#include "containers/static_tensor.hpp"
#include "containers/static_vector.hpp"
#include "generated_config.hpp"
#include "globals/global_config.hpp"
#include "globals/globals.hpp"
#include "tree_builder.hpp"
#include <cassert>

namespace amr::volume
{

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
    template <typename Cell_t, typename Flux_t>
    static auto evaluate_volume_integral(
        Cell_t&                cell_du,
        const Flux_t&          cell_flux,
        [[maybe_unused]] auto& cell_volume,
        auto&                  inverse_jacobian
    )
    {
        for (std::size_t i = 0; i < Policy::Dim; ++i)
        {
            cell_du =
                cell_du + inverse_jacobian * cell_volume *
                              amr::containers::algorithms::tensor::derivative_contraction(
                                  global_t::derivative_tensor,
                                  cell_flux[i] * global_t::volume_mass,
                                  i
                              );
        }
        // cell_du = inverse_jacobian * cell_volume *
        //           amr::containers::algorithms::tensor::tensor_dot(
        //               cell_du, global_t::volume_mass
        //           ); // TODO add the inverse Jacobian, see the volume mass
        // TODO multiply the matrix beforehand
    }
};

} // namespace amr::volume
