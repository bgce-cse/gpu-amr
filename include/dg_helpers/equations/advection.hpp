#ifndef ADVECTION_HPP
#define ADVECTION_HPP

#include "equation_impl.hpp"
#include <cmath>

namespace amr::equations {

/**
 * @brief 2D Linear advection equation.
 * 
 * Inherits from EquationImpl and implements the advection equation:
 * ∂U/∂t + ∂(v*U)/∂x + ∂(v*U)/∂y = 0
 * 
 * where v is uniform velocity in all directions.
 * 
 * @tparam NumDOFs Number of degrees of freedom (conservation variables)
 * @tparam Scalar Floating-point type
 */
template<unsigned int NumDOFs, unsigned int Order, unsigned int Dim, typename Scalar = double>
class Advection2D : public EquationBaseTypes<EquationImpl<NumDOFs, Order, Dim, Scalar>> {

private:
    Scalar velocity_;
    
public:
    /**
     * @brief Constructor with optional velocity parameter.
     * 
     * @param velocity Advection velocity (default: 1.0)
     */
    explicit Advection2D(Scalar velocity = 1.0) : velocity_(velocity) {}
    
    /**
     * @brief Set the advection velocity.
     */
    void set_velocity(Scalar v) noexcept {
        velocity_ = v;
    }
    
    /**
     * @brief Get the advection velocity.
     */
    Scalar get_velocity() const noexcept {
        return velocity_;
    }
    
    // Use "using" declarations to bring dependent types into scope
    using base_t = EquationBaseTypes<EquationImpl<NumDOFs, Order, Dim, Scalar>>;
    using typename base_t::dof_t;
    using typename base_t::flux_t;
    using typename base_t::dof_value_t;
    
    /**
     * @brief Evaluate flux F(U) at a cell in 2D.
     * 
     * For linear advection: F = velocity * U
     * Stores as: cellflux[row 0] = F_x, cellflux[row 1] = F_y
     * 
     * @param celldofs DOF values at cell (NumDOFs elements)
     * @param[out] cellflux Flux in both directions (2 × NumDOFs matrix)
     *                      cellflux[0, j] = F_x[j]
     *                      cellflux[1, j] = F_y[j]
     */
    void evaluate_flux(const dof_t& celldofs, flux_t& cellflux) const override {
        // For each spatial dimension, flux[dim] = velocity_ * celldofs
        for (unsigned int dim = 0; dim < Dim; ++dim) {
            // Iterate over all basis points in the tensor
            auto multi_idx = typename dof_t::multi_index_t{};
            while (true) {
                cellflux[dim][multi_idx] = velocity_ * celldofs[multi_idx];
                if (!multi_idx.increment()) break;
            }
        }
    }
    
    /**
     * @brief Compute maximum eigenvalue in a normal direction.
     * 
     * For linear advection, eigenvalue = |velocity| independent of normal direction.
     * 
     * @param celldofs Cell DOF values (not used for linear advection)
     * @param normalidx Normal direction (0=x, 1=y)
     * @return |velocity|
     */
    Scalar max_eigenvalue(
        const dof_t& ,
        unsigned int 
    ) const override {
        // For linear advection, eigenvalue is constant
        return std::abs(velocity_);
    }
    
    /**
     * @brief Get initial values: planar waves.
     * 
     * Initial conditions:
     * - U[0] = sin(2π(x + y - 2*t))  - propagates along (1,1) direction
     * - U[1] = sin(2π(y - t))         - propagates along (0,1) direction
     * - U[2..N] = 1.0                 - constant background
     * 
     * @param position 2D position (x, y)
     * @param t Time
     * @return Initial DOF values
     */
    dof_value_t get_2D_initial_values(
        const amr::containers::static_vector<Scalar, Dim>& position,
        Scalar t = 0.0
    ) const override {
        dof_value_t result{};
        
        const Scalar PI = 3.14159265358979323846;
        const Scalar x = position[0];
        const Scalar y = position[1];
        
        // First mode: diagonal propagation
        if constexpr (NumDOFs >= 1) {
            result[0] = std::sin(2.0 * PI * (x + y - 2.0 * t));
        }
        
        // Second mode: y-direction propagation
        if constexpr (NumDOFs >= 2) {
            result[1] = std::sin(2.0 * PI * (y - t));
        }
        
        // Remaining modes: constant background
        for (unsigned int dof = 2; dof < NumDOFs; ++dof) {
            result[dof] = 1.0;
        }
        
        return result;
    }
};

} // namespace amr::equations

#endif // ADVECTION_HPP
