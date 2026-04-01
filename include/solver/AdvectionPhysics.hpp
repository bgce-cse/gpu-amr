#ifndef ADVECTION_PHYSICS_H
#define ADVECTION_PHYSICS_H

#include "containers/static_vector.hpp"
#include "solver/cell_types.hpp"
#include <algorithm>
#include <cmath>

// --- HIDE CUDA KEYWORDS FROM GCC ---
#ifndef __CUDACC__
#define __host__
#define __device__
#define __forceinline__ inline
#endif
// -----------------------------------

template <int DIM>
class AdvectionPhysics
{
public:
    static constexpr int NVAR = 1; // Only one variable: the scalar concentration (u)

    // TODO: only temporary fix bc the Unit Tests need to read this
    static constexpr double Velocity[3] = { 1.0, 0.5, 0.0 };

    using FieldTags = std::tuple<amr::cell::Scalar>;

    /**
     * @brief For linear advection, conservative and primitive variables are identical.
     * This function is required by the amr_solver interface.
     */
    __host__ __device__ __forceinline__ static void primitiveToConservative(
        const amr::containers::static_vector<double, NVAR>& prim,
        amr::containers::static_vector<double, NVAR>&       cons,
        [[maybe_unused]] double                             gamma
    )
    {
        cons[0] = prim[0];
    }

    /**
     * @brief Rusanov (Local Lax-Friedrichs) Numerical Flux
     * Used to resolve the state at the interface between two cells.
     */
    __host__ __device__ __forceinline__ static void rusanovFlux(
        const amr::containers::static_vector<double, NVAR>& UL,
        const amr::containers::static_vector<double, NVAR>& UR,
        amr::containers::static_vector<double, NVAR>&       flux,
        int                                                 direction,
        [[maybe_unused]] double                             gamma
    )
    {
        // TODO: fix GPU LOCAL ARRAY: Named 'local_vel' to avoid shadowing warnings
        constexpr double local_vel[3] = { 1.0, 0.5, 0.0 };

        // Calculate physical fluxes
        double fluxL = UL[0] * local_vel[direction];
        double fluxR = UR[0] * local_vel[direction];

        // For linear advection, the wave speed is simply the constant velocity component
        double smax = std::abs(local_vel[direction]);

        // Numerical Flux: F* = 0.5 * (fL + fR) - 0.5 * smax * (uR - uL)
        // This provides upwind-like stability.
        flux[0] = 0.5 * (fluxL + fluxR) - 0.5 * smax * (UR[0] - UL[0]);
    }

    /**
     * @brief Max Speed for CFL condition
     * Returns the maximum characteristic speed in a given direction.
     * Used by amr_solver::compute_time_step to satisfy the CFL condition.
     */
    template <typename PatchTuple>
    __host__ __device__ __forceinline__ static double getMaxSpeed(
        [[maybe_unused]] const PatchTuple& patches, 
        [[maybe_unused]] std::size_t idx, 
        int direction, 
        [[maybe_unused]] double gamma
    ) 
    {
        // TODO: fix GPU LOCAL ARRAY: Named 'local_vel' to avoid shadowing warnings
        constexpr double local_vel[3] = { 1.0, 0.5, 0.0 };
        // The speed is independent of the state in linear advection.
        return std::abs(local_vel[direction]);
    }

    static constexpr int getNumVars()
    {
        return NVAR;
    }
};

#endif // ADVECTION_PHYSICS_H
