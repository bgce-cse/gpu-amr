#include "PressureSolver.hpp"
#include "Discretization.hpp"
#include "domain.hpp"
#include <cmath>
#include <iostream>

SOR::SOR(double omega)
    : _omega(omega)
{
}

double SOR::solve(sim_domain& domain) const
{
    auto& tree = domain.tree();
    double pressure_sum = 0.0;
    double totales_volume = 0.0;
    for (std::size_t idx = 0; idx < tree.size(); idx++)
    {
        if (tree.get_cell_type(idx) != cell_type::FLUID)
        {
            continue;
        }
        auto morton_idx = tree.get_node_index_at(idx);

        double dx = domain.dx(morton_idx);
        double dy = domain.dy(morton_idx);

        double coeff = _omega / (2.0 * (1.0 / (dx * dx) + 1.0 / (dy * dy))
                                ); // = _omega * h^2 / 4.0, if dx == dy == h

        auto& p_value = domain.template get_value<p_t>(morton_idx);

        p_value = (1.0 - _omega) * p_value +
                  coeff * (Discretization::sor_helper(domain, morton_idx) -
                           domain.template get_value<rhs_t>(morton_idx));
        pressure_sum += p_value * dx * dy;
        totales_volume += dx * dy;
    }
    double pressure_mean = pressure_sum / totales_volume;
    for (std::size_t idx = 0; idx < domain.tree().size(); idx++)
    {
        if (tree.get_cell_type(idx) != cell_type::FLUID)
        {
            continue;
        }
        const auto morton_idx = domain.tree().get_node_index_at(idx);
        auto& p_value = domain.template get_value<p_t>(morton_idx);
        p_value -= pressure_mean;
    }

    double res = 0.0;
    double total_volume = 0.0;
    for (std::size_t idx = 0; idx < tree.size(); idx++)
    {
        if (tree.get_cell_type(idx) != cell_type::FLUID)
        {
            continue;
        }
        auto morton_idx = tree.get_node_index_at(idx);
        double dx = domain.dx(morton_idx);
        double dy = domain.dy(morton_idx);
        auto volume = dx * dy;

        double val = Discretization::laplacian<p_t>(domain, morton_idx) -
                     domain.template get_value<rhs_t>(morton_idx);
        res += volume * val * val;
        total_volume += volume;
    }
    res /= total_volume;
    res = std::sqrt(res);
    return res;
}
