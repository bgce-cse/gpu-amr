#pragma once

#include "data_types.hpp"
#include "domain.hpp"
#include "tree_types.hpp"
// #include "tree.hpp"

/**
 * @brief Class of container and modifier for the physical fields
 *
 */
class Fields
{
  public:
    Fields() = default;

    Fields(
        double nu, double gx, double gy, double dt, double tau, double alpha,
        double beta, bool energy_eq
    );

    /**
     * @brief Calculates the convective and diffusive fluxes in x and y
     * direction based on explicit discretization of the momentum equations
     *
     * @param[in] grid in which the fluxes are calculated
     *
     */
    void calculate_fluxes(sim_domain& domain) const;

    /**
     * @brief Right hand side calculations using the fluxes for the pressure
     * Poisson equation
     *
     * @param[in] grid in which the calculations are done
     *
     */
    void calculate_rs(sim_domain& domain) const;

    /**
     * @brief Velocity calculation using pressure values
     *
     * @param[in] grid in which the calculations are done
     *
     */
    void calculate_velocities(sim_domain& domain) const;

    /**
     * @brief Adaptive step size calculation using x-velocity condition,
     * y-velocity condition and CFL condition
     *
     * @param[in] grid in which the calculations are done
     *
     */
    void calculate_temperatures(sim_domain& domain) const;

    /**
     * @brief Tempaerature values computataion by using a fully explicit
     * method
     *
     * @param[in] grid in which the calculations are done
     *
     */
    double calculate_dt(sim_domain& domain);

    double dt() const;

    bool energy_eq() const;

  private:
    double _nu;
    double _gx;
    double _gy;
    double _dt;
    double _tau;
    double _alpha;
    double _beta;
    bool _energy_eq;
};
