// updateFVM.hpp

#ifndef UPDATE_FVM_HPP
#define UPDATE_FVM_HPP

#include <vector>
#include <tuple>

/**
 * @brief Updates the velocity and pressure fields using the finite volume method for a single time-step.
 *
 * @param u Velocity in x-direction
 * @param v Velocity in y-direction
 * @param p Pressure
 * @param nx Grid points in x-direction
 * @param ny Grid points in y-direction
 * @param dx Grid spacing in x-direction
 * @param dy Grid spacing in y-direction
 * @param dt Timestep size
 * @param rho Fluid density
 * @param mu Dynamic viscosity
 * @return u_new, v_new, p_new Outputs
 */
 
std::tuple<
    std::vector<std::vector<double>>,
    std::vector<std::vector<double>>,
    std::vector<std::vector<double>>
> updateFVM(
    std::vector<std::vector<double>>& u,
    std::vector<std::vector<double>>& v,
    std::vector<std::vector<double>>& p,
    int nx, int ny,
    double dx, double dy,
    double dt,
    double rho,
    double mu
);

#endif // UPDATE_FVM_HPP

