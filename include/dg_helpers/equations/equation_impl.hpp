#ifndef EQUATION_IMPL_HPP
#define EQUATION_IMPL_HPP

#include "containers/static_vector.hpp"
#include "containers/static_matrix.hpp"
#include "containers/static_tensor.hpp"
#include "dg_helpers/basis.hpp"

namespace amr::equations {

/**
 * @brief Base class for all equation systems in DG methods.
 * 
 * Defines the interface that all equations must implement:
 * - Flux computation
 * - Maximum eigenvalue calculation
 * - Initial conditions
 * 
 * @tparam NumDOFs Number of degrees of freedom (conservation variables)
 * @tparam Scalar Floating-point type
 */
template<unsigned int NumDOFs, unsigned int Order, unsigned int Dim, typename Scalar = double>
class EquationImpl {

protected:
    using dof_value_t = amr::containers::static_vector<Scalar, NumDOFs>;
    using dof_t = typename amr::containers::utils::types::tensor::hypercube_t<dof_value_t, Order, Dim>;
    using flux_t = amr::containers::static_vector<dof_t, Dim>;

public:
    static constexpr unsigned int num_dofs = NumDOFs;
    
    virtual ~EquationImpl() = default;
    
    /**
     * @brief Evaluate flux F(U) at a cell.
     * 
     * For 2D: Returns flux in x and y directions
     * Flux is stored as [F_x; F_y] with NumDOFs components each
     * 
     * @param celldofs DOF values at cell (NumDOFs elements)
     * @param[out] cellflux Flux components (2*NumDOFs elements)
     *                      cellflux[0:NumDOFs] = F_x
     *                      cellflux[NumDOFs:2*NumDOFs] = F_y
     */
    virtual void evaluate_flux(const dof_t& celldofs, flux_t& cellflux) const = 0;
    
    /**
     * @brief Compute maximum eigenvalue in a given normal direction.
     * 
     * Used for CFL stability condition and Rusanov flux computation.
     * 
     * @param celldofs Cell DOF values
     * @param normalidx Normal direction (0=x, 1=y)
     * @return Maximum absolute eigenvalue
     */
    virtual Scalar max_eigenvalue(
        const dof_t& celldofs,
        unsigned int normalidx
    ) const = 0;
    
    /**
     * @brief Get initial values at a 2D position.
     * 
     * @param position 2D position (x, y)
     * @param t Time
     * @return Initial DOF values
     */
    virtual dof_value_t get_2D_initial_values(
        const amr::containers::static_vector<Scalar, Dim>& position,
        Scalar t = 0.0
    ) const = 0;
};

/**
 * @brief Non-templated bridge class to expose dependent types from EquationImpl.
 * 
 * This class allows derived implementations to directly access dof_t, flux_t, and dof_value_t
 * without the need for typename or this-> disambiguation in template contexts.
 * 
 * The types are public so they can be queried by container types like DGCell.
 * 
 * Usage:
 * template<...> class Advection2D : public EquationBaseTypes<EquationImpl<...>> { ... }
 * 
 * @tparam EqBase Instantiated EquationImpl with all template parameters bound
 */
template<class EqBase>
class EquationBaseTypes : public EqBase {
public:
    // Expose dependent types publicly for use in container templates
    using dof_t       = typename EqBase::dof_t;
    using flux_t      = typename EqBase::flux_t;
    using dof_value_t = typename EqBase::dof_value_t;
};

} // namespace amr::equations

#endif // EQUATION_IMPL_HPP
