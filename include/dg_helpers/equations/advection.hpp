#ifndef ADVECTION_HPP
#define ADVECTION_HPP

#include "equation_impl.hpp"
#include <cmath>

namespace amr::equations
{

// Forward declaration for dimension-specific initial conditions
template <std::size_t NumDOFs, std::size_t Dim, typename Scalar>
struct AdvectionInitialCondition;

/**
 * @brief Generic linear advection equation.
 */
template <
    std::size_t NumDOFs,
    std::size_t Order,
    std::size_t Dim,
    double      Velocity = 1.0,
    typename Scalar      = double>
struct Advection :
    EquationBase<
        Advection<NumDOFs, Order, Dim, Velocity, Scalar>,
        NumDOFs,
        Order,
        Dim,
        Scalar>
{
    static constexpr Scalar velocity = Velocity;

    static constexpr auto evaluate_flux(const typename Advection::dof_t& celldofs)
    {
        typename Advection::flux_t flux{};
        for (std::size_t d = 0; d < Dim; ++d)
            flux[d] = velocity * celldofs;
        return flux;
    }

    static constexpr Scalar max_eigenvalue(
        [[maybe_unused]] const typename Advection::dof_t& celldofs,
        [[maybe_unused]] std::size_t                      normalidx
    )
    {
        return std::abs(velocity);
    }

    static constexpr typename Advection::dof_value_t get_initial_values(
        const amr::containers::static_vector<Scalar, Dim>& position,
        Scalar                                             t = 0.0
    )
    {
        return AdvectionInitialCondition<NumDOFs, Dim, Scalar>::compute(position, t);
    }
};

// Specialization for dimension-specific initial conditions
template <std::size_t NumDOFs, typename Scalar>
struct AdvectionInitialCondition<NumDOFs, 1, Scalar>
{
    static constexpr auto
        compute(const amr::containers::static_vector<Scalar, 1>& position, Scalar t)
    {
        amr::containers::static_vector<Scalar, NumDOFs> result{};
        constexpr Scalar                                PI = 3.14159265358979323846;

        if constexpr (NumDOFs >= 1) result[0] = std::sin(2.0 * PI * (position[0] - t));
        for (std::size_t i = 1; i < NumDOFs; ++i)
            result[i] = 1.0;

        return result;
    }
};

template <std::size_t NumDOFs, typename Scalar>
struct AdvectionInitialCondition<NumDOFs, 2, Scalar>
{
    static constexpr auto
        compute(const amr::containers::static_vector<Scalar, 2>& position, Scalar t)
    {
        amr::containers::static_vector<Scalar, NumDOFs> result{};
        constexpr Scalar                                PI = 3.14159265358979323846;
        const Scalar                                    x  = position[0];
        const Scalar                                    y  = position[1];

        if constexpr (NumDOFs >= 1) result[0] = std::sin(2.0 * PI * (x + y - 2.0 * t));
        if constexpr (NumDOFs >= 2) result[1] = std::sin(2.0 * PI * (y - t));
        for (std::size_t i = 2; i < NumDOFs; ++i)
            result[i] = 1.0;

        return result;
    }
};

template <std::size_t NumDOFs, typename Scalar>
struct AdvectionInitialCondition<NumDOFs, 3, Scalar>
{
    static constexpr auto
        compute(const amr::containers::static_vector<Scalar, 3>& position, Scalar t)
    {
        amr::containers::static_vector<Scalar, NumDOFs> result{};
        constexpr Scalar                                PI = 3.14159265358979323846;
        const Scalar                                    x  = position[0];
        const Scalar                                    y  = position[1];
        const Scalar                                    z  = position[2];

        if constexpr (NumDOFs >= 1)
            result[0] = std::sin(2.0 * PI * (x + y + z - 3.0 * t));
        if constexpr (NumDOFs >= 2) result[1] = std::sin(2.0 * PI * (y - t));
        if constexpr (NumDOFs >= 3) result[2] = std::sin(2.0 * PI * (z - t));
        for (std::size_t i = 3; i < NumDOFs; ++i)
            result[i] = 1.0;

        return result;
    }
};

} // namespace amr::equations

#endif // ADVECTION_HPP