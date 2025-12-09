#ifndef EULER_HPP
#define EULER_HPP

#include "containers/static_tensor.hpp"
#include "containers/static_vector.hpp"
#include "equation_impl.hpp"
#include <cmath>

namespace amr::equations
{

// Forward declaration for dimension-specific initial conditions
template <std::size_t Dim, typename Scalar>
struct EulerInitialCondition;

/**
 * @brief Compressible Euler equations using CRTP pattern.
 *
 * Conservative variables: [rho*u_1, ..., rho*u_Dim, rho, E]
 * where E is total energy per unit volume.
 *
 * @tparam Order DG polynomial order
 * @tparam Dim Spatial dimension
 * @tparam Scalar Floating-point type
 */
template <unsigned int Order, unsigned int Dim, typename Scalar = double>
struct Euler :
    EquationBase<
        Euler<Order, Dim, Scalar>,
        Dim + 2, // NumDOFs: momentum (Dim) + density (1) + energy (1)
        Order,
        Dim,
        Scalar>
{
    static constexpr unsigned int MOMENTUM_START = 0;
    static constexpr unsigned int DENSITY_IDX    = Dim;
    static constexpr unsigned int ENERGY_IDX     = Dim + 1;
    static constexpr Scalar       DEFAULT_GAMMA  = 1.4;

    Scalar gamma_; // Heat capacity ratio

    // Constructors
    constexpr Euler()
        : gamma_(DEFAULT_GAMMA)
    {
    }

    explicit constexpr Euler(Scalar gamma)
        : gamma_(gamma)
    {
    }

    // Accessors
    constexpr void set_gamma(Scalar g) noexcept
    {
        gamma_ = g;
    }

    constexpr Scalar get_gamma() const noexcept
    {
        return gamma_;
    }

    // --- Thermodynamic primitives ---

    /**
     * @brief Compute pressure from conservative variables.
     */
    constexpr Scalar compute_pressure(const typename Euler::dof_value_t& U) const
    {
        const Scalar rho = U[DENSITY_IDX];
        const Scalar E   = U[ENERGY_IDX];

        Scalar kinetic = 0.0;
        for (unsigned int d = 0; d < Dim; ++d)
        {
            const Scalar mom = U[MOMENTUM_START + d];
            kinetic += 0.5 * mom * mom / rho;
        }

        return (gamma_ - 1.0) * (E - kinetic);
    }

    /**
     * @brief Compute total energy from primitives.
     */
    constexpr Scalar compute_energy(
        const amr::containers::static_vector<Scalar, Dim>& momentum,
        Scalar                                             rho,
        Scalar                                             p
    ) const
    {
        Scalar kinetic = 0.0;
        for (unsigned int d = 0; d < Dim; ++d)
            kinetic += 0.5 * momentum[d] * momentum[d] / rho;

        return p / (gamma_ - 1.0) + kinetic;
    }

    /**
     * @brief Compute velocity vector from conservative variables.
     */
    constexpr amr::containers::static_vector<Scalar, Dim>
        compute_velocity(const typename Euler::dof_value_t& U) const
    {
        const Scalar                                rho = U[DENSITY_IDX];
        amr::containers::static_vector<Scalar, Dim> velocity;

        for (unsigned int d = 0; d < Dim; ++d)
            velocity[d] = U[MOMENTUM_START + d] / rho;

        return velocity;
    }

    // --- Required interface for EquationBase ---

    /**
     * @brief Evaluate flux tensor F(U) in all spatial directions.
     */
    static constexpr auto evaluate_flux(const typename Euler::dof_t& celldofs)
    {
        typename Euler::flux_t cellflux{};

        auto idx = typename Euler::dof_t::multi_index_t{};
        do
        {
            const auto&  U   = celldofs[idx];
            const Scalar rho = U[DENSITY_IDX];
            const Scalar E   = U[ENERGY_IDX];

            // Need instance for thermodynamic calculations
            // Note: This is a limitation of making methods static
            // Consider making gamma a template parameter if performance is critical
            Euler        eq; // Uses default gamma
            const Scalar p        = eq.compute_pressure(U);
            const auto   velocity = eq.compute_velocity(U);

            // Compute flux in each spatial direction
            for (unsigned int dir = 0; dir < Dim; ++dir)
            {
                const Scalar u_dir = velocity[dir];

                // Momentum fluxes: rho * u_i * u_dir + p * delta_{i,dir}
                for (unsigned int i = 0; i < Dim; ++i)
                {
                    cellflux[dir][idx][MOMENTUM_START + i] =
                        U[MOMENTUM_START + i] * u_dir;
                    if (i == dir) cellflux[dir][idx][MOMENTUM_START + i] += p;
                }

                // Density flux: rho * u_dir
                cellflux[dir][idx][DENSITY_IDX] = rho * u_dir;

                // Energy flux: (E + p) * u_dir
                cellflux[dir][idx][ENERGY_IDX] = (E + p) * u_dir;
            }
        } while (idx.increment());

        return cellflux;
    }

    /**
     * @brief Compute maximum eigenvalue (wave speed) in given direction.
     */
    static constexpr Scalar
        max_eigenvalue(const typename Euler::dof_t& celldofs, unsigned int normalidx)
    {
        // Evaluate at first quadrature point (can be extended to search all points)
        auto        idx = typename Euler::dof_t::multi_index_t{};
        const auto& U   = celldofs[idx];

        Euler        eq; // Uses default gamma
        const Scalar rho = U[DENSITY_IDX];
        const Scalar p   = eq.compute_pressure(U);
        const Scalar c   = std::sqrt(eq.gamma_ * p / rho);      // Sound speed
        const Scalar u_n = U[MOMENTUM_START + normalidx] / rho; // Normal velocity

        return std::abs(u_n) + c; // Maximum eigenvalue
    }

    /**
     * @brief Get initial condition at given position and time.
     */
    static constexpr typename Euler::dof_value_t get_initial_values(
        const amr::containers::static_vector<Scalar, Dim>& position,
        Scalar                                             t = 0.0
    )
    {
        return EulerInitialCondition<Dim, Scalar>::compute(position, t);
    }
};

// --- Dimension-specific initial conditions ---

/**
 * @brief 1D Euler initial condition: Gaussian pressure pulse.
 */
template <typename Scalar>
struct EulerInitialCondition<1, Scalar>
{
    static constexpr auto compute(
        const amr::containers::static_vector<Scalar, 1>& position,
        [[maybe_unused]] Scalar                          t
    )
    {
        constexpr unsigned int                          NumDOFs = 3; // [rho*u, rho, E]
        amr::containers::static_vector<Scalar, NumDOFs> U{};

        // Gaussian pressure perturbation
        const Scalar x   = position[0];
        const Scalar p   = std::exp(-100.0 * (x - 0.5) * (x - 0.5)) + 1.0;
        const Scalar rho = 1.0;

        U[0] = 0.0; // momentum (rho*u)
        U[1] = rho; // density

        // Compute energy (assuming zero velocity)
        Euler<0, 1, Scalar>                       eq;
        amr::containers::static_vector<Scalar, 1> momentum{ 0.0 };
        U[2] = eq.compute_energy(momentum, rho, p);

        return U;
    }
};

/**
 * @brief 2D Euler initial condition: Gaussian pressure pulse.
 */
template <typename Scalar>
struct EulerInitialCondition<2, Scalar>
{
    static constexpr auto compute(
        const amr::containers::static_vector<Scalar, 2>& position,
        [[maybe_unused]] Scalar                          t
    )
    {
        constexpr unsigned int NumDOFs = 4; // [rho*u, rho*v, rho, E]
        amr::containers::static_vector<Scalar, NumDOFs> U{};

        // Gaussian pressure perturbation centered at (0.5, 0.5)
        const Scalar x   = position[0];
        const Scalar y   = position[1];
        const Scalar r2  = (x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5);
        const Scalar p   = std::exp(-100.0 * r2) + 1.0;
        const Scalar rho = 1.0;

        U[0] = 0.0; // x-momentum
        U[1] = 0.0; // y-momentum
        U[2] = rho; // density

        // Compute energy
        Euler<0, 2, Scalar>                       eq;
        amr::containers::static_vector<Scalar, 2> momentum{ 0.0, 0.0 };
        U[3] = eq.compute_energy(momentum, rho, p);

        return U;
    }
};

/**
 * @brief 3D Euler initial condition: Gaussian pressure pulse.
 */
template <typename Scalar>
struct EulerInitialCondition<3, Scalar>
{
    static constexpr auto compute(
        const amr::containers::static_vector<Scalar, 3>& position,
        [[maybe_unused]] Scalar                          t
    )
    {
        constexpr unsigned int NumDOFs = 5; // [rho*u, rho*v, rho*w, rho, E]
        amr::containers::static_vector<Scalar, NumDOFs> U{};

        // Gaussian pressure perturbation centered at (0.5, 0.5, 0.5)
        const Scalar x = position[0];
        const Scalar y = position[1];
        const Scalar z = position[2];
        const Scalar r2 =
            (x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5) + (z - 0.5) * (z - 0.5);
        const Scalar p   = std::exp(-100.0 * r2) + 1.0;
        const Scalar rho = 1.0;

        U[0] = 0.0; // x-momentum
        U[1] = 0.0; // y-momentum
        U[2] = 0.0; // z-momentum
        U[3] = rho; // density

        // Compute energy
        Euler<0, 3, Scalar>                       eq;
        amr::containers::static_vector<Scalar, 3> momentum{ 0.0, 0.0, 0.0 };
        U[4] = eq.compute_energy(momentum, rho, p);

        return U;
    }
};

} // namespace amr::equations

#endif // EULER_HPP