#ifndef EULER_PHYSICS_H
#define EULER_PHYSICS_H

#include "cell_types.hpp"
#include "containers/static_vector.hpp"
#include <algorithm>
#include <cmath>

template <int DIM>
class EulerPhysics
{
public:
    static constexpr int NVAR = DIM + 2; // Nvariables: density + DIM velocities + energy

    using FieldTags = std::conditional_t<
        DIM == 2,
        std::tuple<amr::cell::Rho, amr::cell::Rhou, amr::cell::Rhov, amr::cell::E2D>,
        std::tuple<
            amr::cell::Rho,
            amr::cell::Rhou,
            amr::cell::Rhov,
            amr::cell::Rhow,
            amr::cell::E3D>>;

    /**
     * @brief Convert conservative to primitive variables
     * @param cons Conservative variables [rho, rho*u, rho*v, (rho*w), E]
     * @param prim Output primitive variables [rho, u, v, (w), p]
     * @param gamma Specific heat ratio
     */
    static void conservativeToPrimitive(
        const amr::containers::static_vector<double, NVAR>& cons,
        amr::containers::static_vector<double, NVAR>&       prim,
        double                                              gamma
    )
    {
        const double rho = cons[0];

        // Density (rho)
        prim[0] = rho;

        // Velocities (u)
        double vel_squared = 0.0;
        for (int dim = 0; dim < DIM; dim++)
        {
            prim[1 + dim] = cons[1 + dim] / rho;
            vel_squared += prim[1 + dim] * prim[1 + dim];
        }

        // Pressure (p)
        const double E = cons[DIM + 1];
        prim[DIM + 1] =
            (gamma - 1.0) *
            (E - 0.5 * rho * vel_squared); // ((gamma-1) * (E - 0.5*rho*|u|^2))
    }

    /**
     * @brief Convert primitive to conservative variables
     * @param prim Primitive variables [rho, u, v, (w), p]
     * @param cons Output conservative variables [rho, rho*u, rho*v, (rho*w), E]
     * @param gamma Specific heat ratio
     */
    static void primitiveToConservative(
        const amr::containers::static_vector<double, NVAR>& prim,
        amr::containers::static_vector<double, NVAR>&       cons,
        double                                              gamma
    )
    {
        double rho = prim[0];
        double p   = prim[DIM + 1];

        // Density (rho)
        cons[0] = rho;

        // Momentum and compute velocity squared (rho*u)
        double vel_squared = 0.0;
        for (int dim = 0; dim < DIM; dim++)
        {
            double u      = prim[1 + dim];
            cons[1 + dim] = rho * u;
            vel_squared += u * u;
        }

        // Total energy (E)
        cons[DIM + 1] =
            p / (gamma - 1.0) + 0.5 * rho * vel_squared; // (p/(gamma-1) + 0.5*rho*|u|^2)
    }

    /**
     * @brief Compute flux in a given direction
     * @param cons Conservative variables
     * @param prim Primitive variables (precalculated)
     * @param flux Output flux vector
     * Flux in direction d:
     * F_d = [rho*u_d, rho*u_d*u_x + p*delta_dx, rho*u_d*u_y + p*delta_dy,
     *        (rho*u_d*u_z + p*delta_dz), u_d*(E + p)]
     * @param direction Direction index (0=x, 1=y, 2=z)
     */
    static void computeFlux(
        const amr::containers::static_vector<double, NVAR>& cons,
        const amr::containers::static_vector<double, NVAR>& prim,
        amr::containers::static_vector<double, NVAR>&       flux,
        int                                                 direction
    )
    {
        double rho   = prim[0];
        double p     = prim[DIM + 1];
        double u_dir = prim[1 + direction]; // velocity in flux direction

        // Mass flux (rho * u)
        flux[0] = rho * u_dir;

        // Momentum flux (rho*u*v, if direction rho*u^2 + p)
        for (int dim = 0; dim < DIM; dim++)
        {
            flux[1 + dim] = rho * u_dir * prim[1 + dim]; // (rho*u*v)
        }
        flux[1 + direction] += p; // Pressure term only in flux direction (p)
        // (rho*u_d*u_i + p*delta_di)

        // Energy flux (u * (E + p))
        flux[DIM + 1] = u_dir * (cons[DIM + 1] + p);
    }

    /**
     * @brief Calculate speed of sound
     * @param prim Primitive variables (precalculated)
     * @param gamma Specific heat ratio
     * @return Sound speed a = sqrt(gamma * p / rho)
     */
    static double computeSoundSpeed(
        const amr::containers::static_vector<double, NVAR>& prim,
        double                                              gamma
    )

    {
        // TODO: Theses accesses should be std::get<Rho>(prim) and std::get<P>(prim)
        const double rho = prim[0];
        const double p   = prim[DIM + 1];
        return std::sqrt(gamma * p / rho);
    }

    /**
     * @brief Rusanov (Local Lax-Friedrichs) Riemann solver - solves Riemann problem at
     * cell interfaces
     * @param UL Left state (conservative)
     * @param UR Right state (conservative)
     * @param flux Output numerical flux
     * @param direction Direction index (0=x, 1=y, 2=z)
     * @param gamma Specific heat ratio
     */
    static void rusanovFlux(
        const amr::containers::static_vector<double, NVAR>& UL,
        const amr::containers::static_vector<double, NVAR>& UR,
        amr::containers::static_vector<double, NVAR>&       flux,
        int                                                 direction,
        double                                              gamma
    )
    {
        // Calculate Primitive variables
        amr::containers::static_vector<double, NVAR> primL, primR;
        conservativeToPrimitive(UL, primL, gamma);
        conservativeToPrimitive(UR, primR, gamma);

        // Compute physical fluxes for left and right states
        amr::containers::static_vector<double, NVAR> fluxL, fluxR;
        computeFlux(UL, primL, fluxL, direction);
        computeFlux(UR, primR, fluxR, direction);

        // Sound speeds
        const double aL = computeSoundSpeed(primL, gamma);
        const double aR = computeSoundSpeed(primR, gamma);

        // Velocity in the flux direction
        const double uL = primL[1 + direction];
        const double uR = primR[1 + direction];

        // Maximum wave speed: smax = max(|u_L| + a_L, |u_R| + a_R)
        const double smax = std::max(std::abs(uL) + aL, std::abs(uR) + aR);

        // Rusanov flux: F* = 0.5*(FL + FR) - 0.5*smax*(UR - UL)
        for (int k = 0; k < NVAR; k++)
        {
            // TODO: Can the two multiplications by 0.5 be changed by just one?
            // Solution seems to change slighlty so I cannot do it immediately
            // This also means that the compiler is issuing both, meaning that
            // merging them would save one multiplication
            flux[k] = 0.5 * (fluxL[k] + fluxR[k]) - 0.5 * smax * (UR[k] - UL[k]);
        }
    }

    /**
     * @brief SoA-compatible Rusanov Flux wrapper
     * Reads directly from a tuple of patch arrays into local registers,
     * then defers to the original rusanovFlux logic.
     */
    template <typename PatchTuple>
    static void rusanovFluxSoA(
        const PatchTuple& patches, 
        std::size_t idx_L, 
        std::size_t idx_R, 
        amr::containers::static_vector<double, NVAR>& flux, 
        int direction, 
        double gamma
    ) 
    {
        // Thread-local storage (maps directly to fast GPU registers)
        amr::containers::static_vector<double, NVAR> UL;
        amr::containers::static_vector<double, NVAR> UR;
        
        // Coalesced read from Global Memory (patches) into Registers (UL, UR)
        auto fetch_state = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            ((UL[Is] = std::get<Is>(patches)[idx_L]), ...);
            ((UR[Is] = std::get<Is>(patches)[idx_R]), ...);
        };
        fetch_state(std::make_index_sequence<NVAR>{});
        rusanovFlux(UL, UR, flux, direction, gamma);
    }

    /**
     * @brief Get the maximum wave speed for the CFL condition
     * @param cons Conservative variables
     * @param direction Direction index
     * @param gamma Specific heat ratio
     * @return Max speed |u| + a
     */
    static double getMaxSpeed(
        const amr::containers::static_vector<double, NVAR>& cons,
        int                                                 direction,
        double                                              gamma
    )
    {
        amr::containers::static_vector<double, NVAR> prim;
        conservativeToPrimitive(cons, prim, gamma);

        double u_dir = prim[1 + direction]; // Velocity in the chosen direction
        double a     = computeSoundSpeed(prim, gamma);

        return std::abs(u_dir) + a;
    }

    /**
     * @brief SoA-compatible wrapper for max wave speed calculation.
     * Reads directly from a tuple of patch arrays into local registers.
     */
    template <typename PatchTuple>
    static double getMaxSpeedSoA(
        const PatchTuple& patches, 
        std::size_t idx, 
        int direction, 
        double gamma
    ) 
    {
        // Read directly from global memory into fast thread-local registers
        amr::containers::static_vector<double, NVAR> cons;
        auto fetch_state = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            ((cons[Is] = std::get<Is>(patches)[idx]), ...);
        };
        fetch_state(std::make_index_sequence<NVAR>{});
        return getMaxSpeed(cons, direction, gamma);
    }


    /**
     * @brief Get number of variables
     */
    static constexpr int getNumVars()
    {
        return NVAR;
    }
};

// Typedefs
using EulerPhysics2D = EulerPhysics<2>;
using EulerPhysics3D = EulerPhysics<3>;

// Direction namespace
namespace Direction
{
constexpr int X = 0;
constexpr int Y = 1;
constexpr int Z = 2;
} // namespace Direction

#endif // EULER_PHYSICS_H
