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

} // namespace amr::equations

#endif // DG_HELPERS_EQUATIONS_HPP