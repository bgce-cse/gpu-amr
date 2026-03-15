#ifndef ADVECTION_HPP
#define ADVECTION_HPP

#include "equation_impl.hpp"
#include <cmath>

namespace amr::equations
{

// Forward declaration for dimension-specific initial conditions
template <std::size_t NumDOF, std::size_t Dim>
struct AdvectionInitialCondition;

/**
 * @brief Generic linear advection equation.
 */
template <std::size_t NumDOF, std::size_t Order, std::size_t Dim, double Velocity = 1.0>
struct Advection :
    EquationBase<Advection<NumDOF, Order, Dim, Velocity>, NumDOF, Order, Dim, double>
{
    static constexpr double velocity = Velocity;

    static constexpr auto evaluate_flux(typename Advection::dof_t celldofs)
    {
        typename Advection::flux_t flux{};
        for (std::size_t d = 0; d < Dim; ++d)
            flux[d] = velocity * celldofs;
        return flux;
    }

    static constexpr double max_eigenvalue(
        [[maybe_unused]] typename Advection::dof_t celldofs,
        [[maybe_unused]] std::integral auto&       normalidx
    )
    {
        return std::abs(velocity);
    }

    static constexpr auto get_initial_values(const auto& position, double t = 0.0)
        -> amr::containers::static_vector<double, NumDOF>
    {
        // Explicitly return the correct type to avoid type deduction issues
        return AdvectionInitialCondition<NumDOF, Dim>::compute(position, t);
    }
};

// Specialization for dimension-specific initial conditions
template <std::size_t NumDOF>
struct AdvectionInitialCondition<NumDOF, 1>
{
    static constexpr auto compute(const auto& position, double t)
        -> amr::containers::static_vector<double, NumDOF>
    {
        amr::containers::static_vector<double, NumDOF> result{};
        constexpr double                               PI = 3.14159265358979323846;

        if constexpr (NumDOF >= 1) result[0] = std::sin(2.0 * PI * (position[0] - t));
        for (std::size_t i = 1; i < NumDOF; ++i)
            result[i] = 1.0;

        return result;
    }
};

template <std::size_t NumDOF>
struct AdvectionInitialCondition<NumDOF, 2>
{
    static constexpr auto compute(const auto& position, double t)
        -> amr::containers::static_vector<double, NumDOF>
    {
        amr::containers::static_vector<double, NumDOF> result{};
        constexpr double                               PI = 3.14159265358979323846;
        const double                                   x  = position[0];
        const double                                   y  = position[1];

        if constexpr (NumDOF >= 1) result[0] = std::sin(2.0 * PI * (x + y - 2.0 * t));
        if constexpr (NumDOF >= 2) result[1] = std::sin(2.0 * PI * (y - t));
        for (std::size_t i = 2; i < NumDOF; ++i)
            result[i] = 1.0;

        return result;
    }
};

template <std::size_t NumDOF>
struct AdvectionInitialCondition<NumDOF, 3>
{
    static constexpr auto compute(const auto& position, double t)
        -> amr::containers::static_vector<double, NumDOF>
    {
        amr::containers::static_vector<double, NumDOF> result{};
        constexpr double                               PI = 3.14159265358979323846;
        const double                                   x  = position[0];
        const double                                   y  = position[1];
        const double                                   z  = position[2];

        if constexpr (NumDOF >= 1) result[0] = std::sin(2.0 * PI * (x + y + z - 3.0 * t));
        if constexpr (NumDOF >= 2) result[1] = std::sin(2.0 * PI * (y - t));
        if constexpr (NumDOF >= 3) result[2] = std::sin(2.0 * PI * (z - t));
        for (std::size_t i = 3; i < NumDOF; ++i)
            result[i] = 1.0;

        return result;
    }
};

} // namespace amr::equations

#endif // ADVECTION_HPP