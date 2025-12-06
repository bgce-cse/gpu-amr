#ifndef DG_HELPERS_EQUATIONS_HPP
#define DG_HELPERS_EQUATIONS_HPP

#include "equations/advection.hpp"
#include "equations/equation_impl.hpp"
#include "equations/euler.hpp"
#include "globals.hpp"
#include "scenario.hpp"
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

namespace amr::equations
{

/**
 * @brief Get number of variables (DOFs - parameters)
 *
 * For most equations, parameters = 0, so this equals DOFs.
 * Override via specialization if parameters exist.
 */
template <typename Equation>
constexpr unsigned int get_nvars(const Equation& eq)
{
    return Equation::num_dofs;
}

/**
 * @brief Get number of material parameters
 *
 * Override via specialization for equations with parameters.
 */
template <typename Equation>
constexpr unsigned int get_nparams(const Equation& /* eq */)
{
    return 0;
}

/**
 * @brief Helper struct to interpolate initial solution values onto DG basis
 *
 * Holds references to the equation and basis, which are initialized once
 * and reused for multiple interpolations.
 *
 * Template parameters:
 * - EquationType: The equation type (provides initial conditions)
 * - BasisType: The DG basis type (provides quadrature points for DOF nodes)
 */
template <typename EquationType, typename BasisType>
struct InitialDOFInterpolator
{
    const EquationType& eq;
    const BasisType&    basis;

    /**
     * @brief Interpolate initial solution values onto DG basis
     *
     * Evaluates initial conditions at the global coordinates of DOF nodes
     * and projects them onto the reference basis.
     *
     * @param cell_coords Global coordinates of cell center (static_vector)
     * @param cell_size Size of the cell in physical domain
     * @return Tensor of DOF values (dof_t type)
     */
    inline auto operator()(const auto& cell_center, double cell_size) const
    {
        return basis.project_to_reference_basis(
            [&](auto node_coords)
            {
                // node_coords is in [0,1] reference space
                // Shift to [-0.5, 0.5] then scale by cell_size
                auto shifted = node_coords;
                for (unsigned int d = 0; d < 2; ++d)
                {
                    shifted[d] -= 0.5;
                }
                const auto& global_pos =
                    amr::global::reference_to_global(cell_center, shifted, cell_size);
                return eq.get_2D_initial_values(global_pos, 0.0);
            }
        );
    }
};

/**
 * @brief Deduction guide for InitialDOFInterpolator
 */
template <typename EquationType, typename BasisType>
InitialDOFInterpolator(const EquationType&, const BasisType&)
    -> InitialDOFInterpolator<EquationType, BasisType>;

} // namespace amr::equations

#endif // DG_HELPERS_EQUATIONS_HPP