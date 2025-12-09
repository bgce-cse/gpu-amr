#ifndef EQUATION_IMPL_HPP
#define EQUATION_IMPL_HPP

#include "containers/static_matrix.hpp"
#include "containers/static_tensor.hpp"
#include "containers/static_vector.hpp"
#include <type_traits>

namespace amr::equations
{

/**
 * @brief Base template for all DG equation systems using CRTP.
 *
 * @tparam Derived The derived equation class
 * @tparam NumDOFs Number of DOFs
 * @tparam Order Polynomial order
 * @tparam Dim Spatial dimension
 * @tparam Scalar Floating-point type
 */
template <
    typename Derived,
    unsigned int NumDOFs,
    unsigned int Order,
    unsigned int Dim,
    typename Scalar = double>
struct EquationBase
{
    using dof_value_t = amr::containers::static_vector<Scalar, NumDOFs>;
    using dof_t       = typename amr::containers::utils::types::tensor::
        hypercube_t<dof_value_t, Order, Dim>;
    using flux_t = amr::containers::static_vector<dof_t, Dim>;

    static constexpr unsigned int num_dofs = NumDOFs;

    // Interface methods that forward to Derived
    static constexpr auto evaluate_flux(const dof_t& celldofs)
    {
        return Derived::evaluate_flux(celldofs);
    }

    static constexpr Scalar max_eigenvalue(const dof_t& celldofs, unsigned int normalidx)
    {
        return Derived::max_eigenvalue(celldofs, normalidx);
    }

    static constexpr dof_value_t get_initial_values(
        const amr::containers::static_vector<Scalar, Dim>& position,
        Scalar                                             t = 0.0
    )
    {
        return Derived::get_initial_values(position, t);
    }

protected:
    // Prevent direct instantiation
    EquationBase() = default;

private:
    // Compile-time interface checking
    friend Derived;

    // Helper to check if Derived implements required methods
    template <typename T, typename = void>
    struct has_required_interface : std::false_type
    {
    };

    template <typename T>
    struct has_required_interface<
        T,
        std::void_t<
            decltype(T::evaluate_flux(std::declval<dof_t>())),
            decltype(T::max_eigenvalue(
                std::declval<dof_t>(),
                std::declval<unsigned int>()
            )),
            decltype(T::get_initial_values(
                std::declval<amr::containers::static_vector<Scalar, Dim>>(),
                std::declval<Scalar>()
            ))>> : std::true_type
    {
    };

    static_assert(
        has_required_interface<Derived>::value || std::is_same_v<Derived, void>,
        "Derived class must implement: evaluate_flux, max_eigenvalue, get_initial_values"
    );
};

} // namespace amr::equations

#endif // EQUATION_IMPL_HPP