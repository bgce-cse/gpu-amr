#ifndef ADVECTION_PHYSICS_H
#define ADVECTION_PHYSICS_H

#include "containers/static_vector.hpp"
#include "solver/cell_types.hpp"
#include <cmath>
#include <algorithm>

template<int DIM>
class AdvectionPhysics {
public:
    static constexpr int NVAR = 1; // Only one variable: the scalar concentration (u)
    
    // Define a constant velocity for the "wind"
    static constexpr double Velocity[3] = {1.0, 0.5, 0.0};

    using FieldTags = std::tuple<amr::cell::Scalar>;

    /**
     * @brief For linear advection, conservative and primitive variables are identical.
     * This function is required by the amr_solver interface.
     */
    static void primitiveToConservative(const amr::containers::static_vector<double, NVAR>& prim, 
                                       amr::containers::static_vector<double, NVAR>& cons, 
                                       [[maybe_unused]] double gamma) {
        cons[0] = prim[0];
    }

    static void conservativeToPrimitive(const amr::containers::static_vector<double, NVAR>& cons, 
                                       amr::containers::static_vector<double, NVAR>& prim, 
                                       [[maybe_unused]] double gamma) {
        prim[0] = cons[0];
    }

    static void computeFlux(const amr::containers::static_vector<double, NVAR>& cons, 
                           amr::containers::static_vector<double, NVAR>& flux, 
                           int direction) {
        flux[0] = cons[0] * Velocity[direction];
    }

    /**
     * @brief Rusanov (Local Lax-Friedrichs) Numerical Flux
     * Used to resolve the state at the interface between two cells.
     */
    template <typename PatchTuple>
    static void rusanovFlux(const PatchTuple& patches, 
        std::size_t idx_L, 
        std::size_t idx_R, 
        amr::containers::static_vector<double, NVAR>& flux, 
        int direction,
        [[maybe_unused]] double gamma) 
    {
        // Thread-local registers
        double UL_scalar = std::get<0>(patches)[idx_L];
        double UR_scalar = std::get<0>(patches)[idx_R];
        
        // Calculate physical fluxes
        double fluxL = UL_scalar * Velocity[direction];
        double fluxR = UR_scalar * Velocity[direction];
        
        // For linear advection, the wave speed is simply the constant velocity component
        double smax = std::abs(Velocity[direction]);
        
        // Numerical Flux: F* = 0.5 * (fL + fR) - 0.5 * smax * (uR - uL)
        // This provides upwind-like stability.
        flux[0] = 0.5 * (fluxL + fluxR) - 0.5 * smax * (UR_scalar - UL_scalar);
    }

    /**
     * @brief SoA-compatible wrapper for max wave speed calculation.
     */
    template <typename PatchTuple>
    static double getMaxSpeedSoA([[maybe_unused]] const PatchTuple& patches, 
                                 [[maybe_unused]] std::size_t idx, 
                                 int direction, 
                                 [[maybe_unused]] double gamma) {
        
        // The speed is independent of the state U in linear advection
        return std::abs(Velocity[direction]);
    }

    static constexpr int getNumVars() {
        return NVAR;
    }
};

# endif // ADVECTION_PHYSICS_H