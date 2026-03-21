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
     * @brief Rusanov (Local Lax-Friedrichs) Riemann solver - solves Riemann problem at
     * cell interfaces - high performance flattened version
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
        // --- Left State (Calculate completely in registers) ---
        const double inv_rho_L = 1.0 / UL[0];
        double K_L = 0.0;                     // Kinetic energy term
        for (int d = 0; d < DIM; ++d) K_L += UL[1 + d] * UL[1 + d];
        K_L *= 0.5 * inv_rho_L;
        
        const double p_L     = (gamma - 1.0) * (UL[DIM + 1] - K_L);
        const double a_L     = std::sqrt(gamma * p_L * inv_rho_L);
        const double u_dir_L = UL[1 + direction] * inv_rho_L;

        // --- Right State (Calculate completely in registers) ---
        const double inv_rho_R = 1.0 / UR[0]; 
        double K_R = 0.0;                     
        for (int d = 0; d < DIM; ++d) K_R += UR[1 + d] * UR[1 + d];
        K_R *= 0.5 * inv_rho_R;
        
        const double p_R     = (gamma - 1.0) * (UR[DIM + 1] - K_R);
        const double a_R     = std::sqrt(gamma * p_R * inv_rho_R);
        const double u_dir_R = UR[1 + direction] * inv_rho_R;

        // --- Maximum Wave Speed ---
        const double smax = std::max(std::abs(u_dir_L) + a_L, std::abs(u_dir_R) + a_R);

        // --- Flux Calculation ---
        // Mass Flux
        flux[0] = 0.5 * (UL[1 + direction] + UR[1 + direction] - smax * (UR[0] - UL[0]));

        // Momentum Fluxes
        for (int d = 0; d < DIM; ++d)
        {
            // Note: UL[1+d] * u_dir_L is mathematically identical to rho * u * v
            double fL_mom = UL[1 + d] * u_dir_L; 
            double fR_mom = UR[1 + d] * u_dir_R;
            
            // Add pressure to the flux direction
            if (d == direction) 
            {
                fL_mom += p_L;
                fR_mom += p_R;
            }
            flux[1 + d] = 0.5 * (fL_mom + fR_mom - smax * (UR[1 + d] - UL[1 + d]));
        }

        // Energy Flux
        const double fL_E = u_dir_L * (UL[DIM + 1] + p_L);
        const double fR_E = u_dir_R * (UR[DIM + 1] + p_R);
        flux[DIM + 1] = 0.5 * (fL_E + fR_E - smax * (UR[DIM + 1] - UL[DIM + 1]));
    }

    /**
     * @brief Get the maximum wave speed for the CFL condition - high performance flattened version
     * Reads directly from a tuple of patch arrays into local registers.
     * @return Max speed |u| + a
     */
    template <typename PatchTuple>
    static double getMaxSpeed(
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

        const double inv_rho = 1.0 / cons[0];
        double K = 0.0;
        for (int d = 0; d < DIM; ++d) K += cons[1 + d] * cons[1 + d];
        K *= 0.5 * inv_rho;

        const double p     = (gamma - 1.0) * (cons[DIM + 1] - K);
        const double a     = std::sqrt(gamma * p * inv_rho);
        const double u_dir = cons[1 + direction] * inv_rho;

        return std::abs(u_dir) + a;
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
