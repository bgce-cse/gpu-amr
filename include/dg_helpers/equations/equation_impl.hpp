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
 * @tparam NumDOF Number of DOFs
 * @tparam Order Polynomial order
 * @tparam Dim Spatial dimension
 * @tparam Scalar Floating-point type
 */
template <
    typename Derived,
    std::size_t NumDOF,
    std::size_t Order,
    std::size_t Dim,
    typename Scalar = double>
struct EquationBase
{
    using dof_value_t = amr::containers::static_vector<Scalar, NumDOF>;
    using dof_t       = typename amr::containers::utils::types::tensor::
        hypercube_t<dof_value_t, Order, Dim>;
    using flux_t = amr::containers::static_vector<dof_t, Dim>;

    static constexpr std::size_t num_dofs = NumDOF;

    // Interface methods that forward to Derived
    static constexpr auto evaluate_flux(const dof_t& celldofs)
    {
        return Derived::evaluate_flux(celldofs);
    }

    static constexpr Scalar max_eigenvalue(const dof_t& celldofs, std::size_t normalidx)
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

public:
    // Allow instantiation for use in evaluators
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
                std::declval<std::size_t>()
            )),
            decltype(T::get_initial_values(
                std::declval<amr::containers::static_vector<Scalar, Dim>>(),
                std::declval<Scalar>()
            ))>> : std::true_type
    {
    };
};

} // namespace amr::equations

#endif // EQUATION_IMPL_HPP