#include "Fields.hpp"
#include "Discretization.hpp"
#include "domain.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
// #include "morton_id.hpp"

Fields::Fields(
    double nu, double gx, double gy, double dt, double tau, double alpha,
    double beta, bool energy_eq
)
    : _nu{nu}
    , _gx{gx}
    , _gy{gy}
    , _dt{dt}
    , _tau{tau}
    , _alpha{alpha}
    , _beta{beta}
    , _energy_eq{energy_eq}
{
}

void Fields::calculate_temperatures(sim_domain& domain) const
{
    auto& tree = domain.tree();

    // Create a vector of the same dimensions but filled with zeros
    std::vector<double> delta_T(tree.size(), 0.0);

    for (std::size_t idx = 0; idx < tree.size(); idx++)
    {
        if (tree.get_cell_type(idx) != cell_type::FLUID)
        {
            continue;
        }
        auto morton_idx = tree.get_node_index_at(idx);

        const auto laplacian_t =
            Discretization::laplacian<t_t>(domain, morton_idx);
        const auto convection_t =
            Discretization::convection_t(domain, morton_idx);

        delta_T[idx] = _dt * (_alpha * laplacian_t - convection_t);
    }

    for (std::size_t idx = 0; idx < tree.size(); idx++)
    {
        if (tree.get_cell_type(idx) != cell_type::FLUID)
        {
            continue;
        }
        auto morton_idx = tree.get_node_index_at(idx);

        auto& t_value = domain.template get_value<t_t>(morton_idx);
        t_value += delta_T[idx];
    }
}

void Fields::calculate_fluxes(sim_domain& domain) const
{
    auto& tree = domain.tree();
    for (std::size_t idx = 0; idx < tree.size(); idx++)
    {
        if (tree.get_cell_type(idx) != cell_type::FLUID)
        {
            continue;
        }
        auto morton_idx = tree.get_node_index_at(idx);

        const auto laplacian_u =
            Discretization::laplacian<u_t>(domain, morton_idx);
        const auto convection_u =
            Discretization::convection_u(domain, morton_idx);

        const auto t_value = domain.template get_value<t_t>(morton_idx);
        const auto buoyancy_x = _energy_eq ? _beta * _dt * (t_value)*_gx : 0;

        auto& f_value = domain.template get_value<f_t>(morton_idx);
        auto u_value = domain.template get_value<u_t>(morton_idx);

        f_value =
            u_value + _dt * (_nu * laplacian_u - convection_u) - buoyancy_x;

        const auto laplacian_v =
            Discretization::laplacian<v_t>(domain, morton_idx);
        const auto convection_v =
            Discretization::convection_v(domain, morton_idx);

        const auto buoyancy_y = _energy_eq ? _beta * _dt * (t_value)*_gy : 0;

        auto& g_value = domain.template get_value<g_t>(morton_idx);
        auto v_value = domain.template get_value<v_t>(morton_idx);

        g_value =
            v_value + _dt * (_nu * laplacian_v - convection_v) - buoyancy_y;
    }
}

void Fields::calculate_rs(sim_domain& domain) const
{
    for (std::size_t idx = 0; idx < domain.tree().size(); idx++)
    {
        if (domain.tree().get_cell_type(idx) != cell_type::FLUID)
        {
            continue;
        }
        const auto morton_idx = domain.tree().get_node_index_at(idx);
        auto& rhs_value = domain.template get_value<rhs_t>(morton_idx);
        const auto left_f_value =
            domain.template get_neighbor_value<f_t>(morton_idx, direction_t::W);
        const auto right_f_value =
            domain.template get_neighbor_value<f_t>(morton_idx, direction_t::E);

        const auto bottom_g_value =
            domain.template get_neighbor_value<g_t>(morton_idx, direction_t::S);
        const auto top_g_value =
            domain.template get_neighbor_value<g_t>(morton_idx, direction_t::N);

        rhs_value = ((right_f_value - left_f_value) / domain.dx(morton_idx) +
                     (top_g_value - bottom_g_value) / domain.dy(morton_idx)) /
                    (_dt * 2);
    }

    double rhs_sum = 0.0;
    double total_volume = 0.0;

    for (std::size_t idx = 0; idx < domain.tree().size(); idx++)
    {
        if (domain.tree().get_cell_type(idx) != cell_type::FLUID)
        {
            continue;
        }
        const auto morton_idx = domain.tree().get_node_index_at(idx);

        auto dx = domain.dx(morton_idx);
        auto dy = domain.dx(morton_idx);
        double volume = dx * dy;
        auto rhs = domain.template get_value<rhs_t>(morton_idx);
        rhs_sum += rhs * volume;
        total_volume += volume;
    }

    double rhs_mean = rhs_sum / total_volume;

    for (std::size_t idx = 0; idx < domain.tree().size(); idx++)
    {
        if (domain.tree().get_cell_type(idx) != cell_type::FLUID)
        {
            continue;
        }
        const auto morton_idx = domain.tree().get_node_index_at(idx);
        auto& rhs = domain.template get_value<rhs_t>(morton_idx);
        rhs -= rhs_mean;
    }
}

void Fields::calculate_velocities(sim_domain& domain) const
{
    for (std::size_t idx = 0; idx < domain.tree().size(); idx++)
    {
        if (domain.tree().get_cell_type(idx) != cell_type::FLUID)
        {
            continue;
        }
        const auto morton_idx = domain.tree().get_node_index_at(idx);

        // U velocity update
        auto& u_value = domain.template get_value<u_t>(morton_idx);
        const auto f_value = domain.template get_value<f_t>(morton_idx);
        const auto right_p_value =
            domain.template get_neighbor_value<p_t>(morton_idx, direction_t::E);
        const auto left_p_value =
            domain.template get_neighbor_value<p_t>(morton_idx, direction_t::W);

        u_value = f_value - _dt * (right_p_value - left_p_value) /
                                domain.dx(morton_idx) / 2;

        // V velocity update
        auto& v_value = domain.template get_value<v_t>(morton_idx);
        const auto g_value = domain.template get_value<g_t>(morton_idx);
        const auto top_p_value =
            domain.template get_neighbor_value<p_t>(morton_idx, direction_t::N);
        const auto bottom_p_value =
            domain.template get_neighbor_value<p_t>(morton_idx, direction_t::S);

        v_value = g_value - _dt * (top_p_value - bottom_p_value) /
                                domain.dy(morton_idx) / 2;
    }
}

double Fields::calculate_dt(sim_domain& domain)
{
    double lim1 = std::numeric_limits<double>::max();
    double lim2 = std::numeric_limits<double>::max();
    double lim3 = std::numeric_limits<double>::max();
    double lim4 = std::numeric_limits<double>::max();

    const auto abs_min = [](auto acc, auto v) {
        return std::min(acc, std::abs(v));
    };

    for (std::size_t idx = 0; idx < domain.tree().size(); idx++)
    {
        if (domain.tree().get_cell_type(idx) != cell_type::FLUID)
        {
            continue;
        }
        const auto morton_idx = domain.tree().get_node_index_at(idx);
        const auto dx = domain.dx(morton_idx);
        const auto dy = domain.dy(morton_idx);
        const auto u_value = domain.template get_value<u_t>(morton_idx);
        const auto v_value = domain.template get_value<v_t>(morton_idx);
        lim1 = abs_min(lim1, 0.5 / _nu / (1.0 / (dx * dx) + 1.0 / (dy * dy)));
        lim2 = abs_min(lim2, dx / u_value);
        lim3 = abs_min(lim3, dy / v_value);
        if (_energy_eq)
        {
            lim4 = abs_min(
                lim4, 0.5 / _alpha / (1.0 / (dx * dx) + 1.0 / (dy * dy))
            );
        }
    }

    auto lim = std::min({lim1, lim2, lim3});
    if (_energy_eq)
    {
        lim = std::min({lim, lim4});
    }

    return _dt = _tau * lim;
}

double Fields::dt() const
{
    return _dt;
}

bool Fields::energy_eq() const
{
    return _energy_eq;
}
