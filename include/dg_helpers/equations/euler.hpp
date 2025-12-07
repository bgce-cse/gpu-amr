#ifndef EULER_HPP
#define EULER_HPP

#include "equation_impl.hpp"
#include "containers/static_vector.hpp"
#include "containers/static_tensor.hpp"
#include <cmath>

namespace amr::equations {

template<unsigned int Order, unsigned int Dim, typename Scalar = double>
class Euler : public EquationBaseTypes<EquationImpl<Dim + 2, Order, Dim, Scalar>> {

    using base_t = EquationBaseTypes<EquationImpl<Dim + 2, Order, Dim, Scalar>>;
    using typename base_t::dof_t;
    using typename base_t::flux_t;
    using typename base_t::dof_value_t;

private:
    Scalar gamma_; // Heat capacity ratio

public:
    explicit Euler(Scalar gamma = 1.4) : gamma_(gamma) {}

    void set_gamma(Scalar g) noexcept { gamma_ = g; }
    Scalar get_gamma() const noexcept { return gamma_; }

    // --- Thermodynamics ---
    Scalar evaluate_pressure(const dof_value_t& U) const {
        const Scalar rho = U[Dim];
        const Scalar E   = U[Dim + 1];

        Scalar kinetic = 0.0;
        for (unsigned int d = 0; d < Dim; ++d)
            kinetic += 0.5 * (U[d] * U[d]) / rho;

        return (gamma_ - 1.0) * (E - kinetic);
    }

    Scalar evaluate_energy(const amr::containers::static_vector<Scalar, Dim>& rhou,
                           Scalar rho, Scalar p) const {
        Scalar kinetic = 0.0;
        for (unsigned int d = 0; d < Dim; ++d)
            kinetic += 0.5 * (rhou[d] * rhou[d]) / rho;
        return p / (gamma_ - 1.0) + kinetic;
    }

    // --- Flux evaluation ---
    void evaluate_flux(const dof_t& celldofs, flux_t& cellflux) const override {
        auto idx = typename dof_t::multi_index_t{};
        while (true) {
            const dof_value_t& U = celldofs[idx];
            const Scalar rho = U[Dim];
            const Scalar E   = U[Dim + 1];
            const Scalar p   = evaluate_pressure(U);

            // Velocity components
            amr::containers::static_vector<Scalar, Dim> u;
            for (unsigned int d = 0; d < Dim; ++d)
                u[d] = U[d] / rho;

            // Fluxes for each direction
            for (unsigned int i = 0; i < Dim; ++i) {
                for (unsigned int j = 0; j < Dim; ++j) {
                    cellflux[i][idx][j] = U[j] * u[i];  // rho u_j u_i
                    if (i == j) cellflux[i][idx][j] += p; // + p δ_ij
                }

                // Density flux
                cellflux[i][idx][Dim] = rho * u[i];
                // Energy flux
                cellflux[i][idx][Dim + 1] = u[i] * (E + p);
            }

            if (!idx.increment()) break;
        }
    }

    // --- Eigenvalue ---
    Scalar max_eigenvalue(const dof_t& celldofs, unsigned int normalidx) const override {
        auto idx = typename dof_t::multi_index_t{};
        const dof_value_t& U = celldofs[idx];

        const Scalar rho = U[Dim];
        const Scalar p   = evaluate_pressure(U);
        const Scalar c   = std::sqrt(gamma_ * p / rho);
        const Scalar un  = U[normalidx] / rho;

        return std::abs(un) + c;
    }

    // --- Example initial condition (Gaussian wave) ---
    dof_value_t get_2D_initial_values(
        const amr::containers::static_vector<Scalar, Dim>& position,
        Scalar /* t */
    ) const override {
        dof_value_t U{};
        Scalar p = std::exp(-100.0 * (position[0] - 0.5) * (position[0] - 0.5)) + 1.0;
        Scalar rho = 1.0;
        
        // Extract momentum components from U for energy calculation
        amr::containers::static_vector<Scalar, Dim> rhou;
        for (unsigned int d = 0; d < Dim; ++d) {
            U[d] = 0.0;
            rhou[d] = 0.0;
        }
        
        U[Dim] = rho;
        U[Dim + 1] = evaluate_energy(rhou, rho, p);
        return U;
    }
};

} // namespace amr::equations

#endif // EULER_HPP
