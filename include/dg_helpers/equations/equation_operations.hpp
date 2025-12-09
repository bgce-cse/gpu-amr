#ifndef DG_HELPERS_EQUATIONS_HPP
#define DG_HELPERS_EQUATIONS_HPP

#include "dg_helpers/globals/globals.hpp"
#include "dg_helpers/scenario.hpp"
#include "equations/advection.hpp"
#include "equations/equation_impl.hpp"
#include "equations/euler.hpp"
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

namespace amr::equations
{

/**
 * @brief Get number of variables (DOFs - parameters)
 */
template <typename Equation>
constexpr unsigned int get_nvars(const Equation& eq)
{
    return Equation::num_dofs;
}

/**
 * @brief Get number of material parameters
 */
template <typename Equation>
constexpr unsigned int get_nparams(const Equation& /* eq */)
{
    return 0;
}

/**
 * @brief Interpolate initial solution onto DG basis
 *
 * Uses basis from GlobalConfig to project initial conditions onto
 * tensor-product Lagrange basis at quadrature points.
 *
 * Template parameters:
 * - GlobalConfig: Configuration with embedded basis and quadrature data
 * - EquationType: The equation type (provides initial conditions)
 */
template <typename GlobalConfigType, typename EquationType>
struct InitialDOFInterpolator
{
    const EquationType& eq;

    using Order = std::integral_constant<unsigned int, GlobalConfigType::Basis_t::order>;
    using Dim =
        std::integral_constant<unsigned int, GlobalConfigType::Basis_t::dimensions>;

    /**
     * @brief Interpolate initial solution onto DG basis
     *
     * Evaluates initial conditions at cell's quadrature point nodes
     * and projects them onto the reference basis.
     *
     * @param cell_center Global coordinates of cell center
     * @param cell_size Size of the cell in physical domain
     * @return Tensor of DOF values
     */
    inline auto operator()(const auto& cell_center, double cell_size) const
    {
        using Basis_t = typename GlobalConfigType::Basis_t;

        return Basis_t::project_to_reference_basis(
            [&](auto node_coords)
            {
                // node_coords is in [0,1] reference space
                // Shift to [-0.5, 0.5] then scale by cell_size
                auto shifted = node_coords;
                for (unsigned int d = 0; d < Dim::value; ++d)
                    shifted[d] -= 0.5;

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
template <typename GlobalConfigType, typename EquationType>
InitialDOFInterpolator(const EquationType&)
    -> InitialDOFInterpolator<GlobalConfigType, EquationType>;

} // namespace amr::equations

#endif // DG_HELPERS_EQUATIONS_HPP