#ifndef EULER_PHYSICS_H
#define EULER_PHYSICS_H

#include "containers/static_vector.hpp"
#include <cmath>
#include <algorithm>

template<int DIM>
class EulerPhysics {
public:
    static constexpr int NVAR = DIM + 2;  // Nvariables: density + DIM velocities + energy
    
    /**
     * @brief Convert conservative to primitive variables
     * @param cons Conservative variables [rho, rho*u, rho*v, (rho*w), E]
     * @param prim Output primitive variables [rho, u, v, (w), p]
     * @param gamma Specific heat ratio
     */
    static void conservativeToPrimitive(const amr::containers::static_vector<double, NVAR>& cons, 
                                       amr::containers::static_vector<double, NVAR>& prim, 
                                       double gamma) {
        double rho = cons[0];
        
        // Density (rho)
        prim[0] = rho;
        
        // Velocities (u)
        double vel_squared = 0.0;
        for (int dim = 0; dim < DIM; dim++) {
            prim[1 + dim] = cons[1 + dim] / rho;
            vel_squared += prim[1 + dim] * prim[1 + dim];
        }
        
        // Pressure (p)
        double E = cons[DIM + 1];
        prim[DIM + 1] = (gamma - 1.0) * (E - 0.5 * rho * vel_squared); // ((gamma-1) * (E - 0.5*rho*|u|^2))
    }
    
    /**
     * @brief Convert primitive to conservative variables
     * @param prim Primitive variables [rho, u, v, (w), p]
     * @param cons Output conservative variables [rho, rho*u, rho*v, (rho*w), E]
     * @param gamma Specific heat ratio
     */
    static void primitiveToConservative(const amr::containers::static_vector<double, NVAR>& prim, amr::containers::static_vector<double, NVAR>& cons, double gamma) {
        double rho = prim[0];
        double p = prim[DIM + 1];
        
        // Density (rho)
        cons[0] = rho;
        
        // Momentum and compute velocity squared (rho*u)
        double vel_squared = 0.0;
        for (int dim = 0; dim < DIM; dim++) {
            double u = prim[1 + dim];
            cons[1 + dim] = rho * u;
            vel_squared += u * u;
        }
        
        // Total energy (E)
        cons[DIM + 1] = p / (gamma - 1.0) + 0.5 * rho * vel_squared; // (p/(gamma-1) + 0.5*rho*|u|^2)
    }
    
    /**
     * @brief Compute flux in a given direction
     * @param cons Conservative variables
     * @param flux Output flux vector
     * Flux in direction d:
     * F_d = [rho*u_d, rho*u_d*u_x + p*delta_dx, rho*u_d*u_y + p*delta_dy, 
     *        (rho*u_d*u_z + p*delta_dz), u_d*(E + p)]
     * @param direction Direction index (0=x, 1=y, 2=z)
     * @param gamma Specific heat ratio
     */
    static void computeFlux(const amr::containers::static_vector<double, NVAR>& cons, amr::containers::static_vector<double, NVAR>& flux, int direction, double gamma) {
        amr::containers::static_vector<double, NVAR> prim;
        conservativeToPrimitive(cons, prim, gamma);
        
        double rho = prim[0];
        double p = prim[DIM + 1];
        double u_dir = prim[1 + direction];  // velocity in flux direction
        
        // Mass flux (rho * u)
        flux[0] = rho * u_dir;
        
        // Momentum flux (rho*u*v, if direction rho*u^2 + p)
        for (int dim = 0; dim < DIM; dim++) {
            flux[1 + dim] = rho * u_dir * prim[1 + dim]; // (rho*u*v)
            if (dim == direction) {
                flux[1 + dim] += p;  // Pressure term only in flux direction (p)
            } // (rho*u_d*u_i + p*delta_di)
        }
        
        // Energy flux (u * (E + p))
        flux[DIM + 1] = u_dir * (cons[DIM + 1] + p);
    }
    
    /**
     * @brief Calculate speed of sound
     * @param cons Conservative variables
     * @param gamma Specific heat ratio
     * @return Sound speed a = sqrt(gamma * p / rho)
     */
    static double computeSoundSpeed(const amr::containers::static_vector<double, NVAR>& cons, double gamma) {
        amr::containers::static_vector<double, NVAR> prim;
        conservativeToPrimitive(cons, prim, gamma);
        double rho = prim[0];
        double p = prim[DIM + 1];
        return std::sqrt(gamma * p / rho);
    }
    
    /**
     * @brief Rusanov (Local Lax-Friedrichs) Riemann solver - solves Riemann problem at cell interfaces
     * @param UL Left state (conservative)
     * @param UR Right state (conservative)
     * @param flux Output numerical flux
     * @param direction Direction index (0=x, 1=y, 2=z)
     * @param gamma Specific heat ratio
     */
    static void rusanovFlux(const amr::containers::static_vector<double, NVAR>& UL, const amr::containers::static_vector<double, NVAR>& UR, amr::containers::static_vector<double, NVAR>& flux, int direction, double gamma) {
        // Compute physical fluxes for left and right states
        amr::containers::static_vector<double, NVAR> fluxL, fluxR;
        computeFlux(UL, fluxL, direction, gamma);
        computeFlux(UR, fluxR, direction, gamma);
        
        // Convert to primitive for wave speed calculation
        amr::containers::static_vector<double, NVAR> primL, primR;
        conservativeToPrimitive(UL, primL, gamma);
        conservativeToPrimitive(UR, primR, gamma);
        
        // Sound speeds
        double aL = computeSoundSpeed(UL, gamma);
        double aR = computeSoundSpeed(UR, gamma);
        
        // Velocity in the flux direction
        double uL = primL[1 + direction];
        double uR = primR[1 + direction];
        
        // Maximum wave speed: smax = max(|u_L| + a_L, |u_R| + a_R)
        double smax = std::max(std::abs(uL) + aL, std::abs(uR) + aR);
        
        // Rusanov flux: F* = 0.5*(FL + FR) - 0.5*smax*(UR - UL)
        for (int k = 0; k < NVAR; k++) {
            flux[k] = 0.5 * (fluxL[k] + fluxR[k]) - 0.5 * smax * (UR[k] - UL[k]);
        }
    }
    
    /**
     * @brief Get number of variables
     */
    static constexpr int getNumVars() {
        return NVAR;
    }
};

// Typedefs
using EulerPhysics2D = EulerPhysics<2>;
using EulerPhysics3D = EulerPhysics<3>;

// Direction namespace
namespace Direction {
    constexpr int X = 0;
    constexpr int Y = 1;
    constexpr int Z = 2;
}

#endif // EULER_PHYSICS_H