#pragma once

#include "Utils.hpp"
#include "data_types.hpp"
#include "domain.hpp"
#include "tree_types.hpp"
// #include "morton_id.hpp"
// #include "tree.hpp"

class Discretization
{
    Discretization() = delete;

  public:
    static void set_gamma(double gamma)
    {
        _gamma = gamma;
    }

    static double convection_u(sim_domain& domain, index_t morton_id)
    {
        const auto u_value = domain.template get_value<u_t>(morton_id);
        const auto u_left =
            domain.template get_neighbor_value<u_t>(morton_id, direction_t::W);
        const auto u_right =
            domain.template get_neighbor_value<u_t>(morton_id, direction_t::E);
        const auto u_top =
            domain.template get_neighbor_value<u_t>(morton_id, direction_t::N);
        const auto u_bottom =
            domain.template get_neighbor_value<u_t>(morton_id, direction_t::S);
        const auto v_value = domain.tree().template get<v_t>(morton_id);
        const auto v_top =
            domain.template get_neighbor_value<v_t>(morton_id, direction_t::N);
        const auto v_bottom =
            domain.template get_neighbor_value<v_t>(morton_id, direction_t::S);

        // refers to d(u^2)/dx
        double convective_term_1_central =
            (utils::pow2(u_value + u_right) - utils::pow2(u_value + u_left)) /
            (4.0 * domain.dx(morton_id));

        const double convective_term_1_artificial_diff =
            (std::abs(u_value + u_right) * (u_value - u_right) -
             std::abs(u_left + u_value) * (u_left - u_value)) /
            (4.0 * domain.dx(morton_id));

        const double du2_dx = convective_term_1_central +
                              _gamma * convective_term_1_artificial_diff;

        // refers to d(uv)/dy
        const double convective_term_2_central =
            ((u_value + u_top) * (v_value + v_top) -
             (u_value + u_bottom) * (v_value + v_bottom)) /
            (4.0 * domain.dy(morton_id));

        const double convective_term_2_artificial_diff =
            (std::abs(v_value + v_top) * (u_value - u_top) -
             std::abs(v_bottom + v_value) * (u_bottom - u_value)) /
            (4.0 * domain.dy(morton_id));

        const double duv_dy = convective_term_2_central +
                              _gamma * convective_term_2_artificial_diff;

        return du2_dx + duv_dy;
    }

    static double convection_v(sim_domain& domain, index_t morton_id)
    {
        const auto v_value = domain.tree().template get<v_t>(morton_id);
        const auto v_left =
            domain.template get_neighbor_value<v_t>(morton_id, direction_t::W);
        const auto v_right =
            domain.template get_neighbor_value<v_t>(morton_id, direction_t::E);
        const auto v_top =
            domain.template get_neighbor_value<v_t>(morton_id, direction_t::N);
        const auto v_bottom =
            domain.template get_neighbor_value<v_t>(morton_id, direction_t::S);

        const auto u_value = domain.tree().get<u_t>(morton_id);
        const auto u_left =
            domain.template get_neighbor_value<u_t>(morton_id, direction_t::W);
        const auto u_right =
            domain.template get_neighbor_value<u_t>(morton_id, direction_t::E);

        // refers to d(v^2)/dy
        const double convective_term_1_central =
            (utils::pow2(v_value + v_top) - utils::pow2(v_value + v_bottom)) /
            (4.0 * domain.dy(morton_id));

        const double convective_term_1_artificial_diff =
            (std::abs(v_value + v_top) * (v_value - v_top) -
             std::abs(v_bottom + v_value) * (v_bottom - v_value)) /
            (4.0 * domain.dy(morton_id));

        const double dv2_dy = convective_term_1_central +
                              _gamma * convective_term_1_artificial_diff;

        // refers to d(uv)/dx
        double convective_term_2_central =
            ((v_value + v_right) * (u_value + u_right) -
             (v_value + v_left) * (u_value + u_left)) /
            (4.0 * domain.dx(morton_id));

        double convective_term_2_artificial_diff =
            (std::abs(u_value + u_right) * (v_value - v_right) -
             std::abs(u_left + u_value) * (v_left - v_value)) /
            (4.0 * domain.dx(morton_id));

        const double duv_dx = convective_term_2_central +
                              _gamma * convective_term_2_artificial_diff;

        return dv2_dy + duv_dx;
    }

    static double convection_t(sim_domain& domain, index_t morton_id)
    {
        const auto t_value = domain.tree().template get<t_t>(morton_id);
        const auto t_left =
            domain.template get_neighbor_value<t_t>(morton_id, direction_t::W);
        const auto t_right =
            domain.template get_neighbor_value<t_t>(morton_id, direction_t::E);
        const auto t_top =
            domain.template get_neighbor_value<t_t>(morton_id, direction_t::N);
        const auto t_bottom =
            domain.template get_neighbor_value<t_t>(morton_id, direction_t::S);

        const auto u_value = domain.tree().get<u_t>(morton_id);
        const auto u_left =
            domain.template get_neighbor_value<u_t>(morton_id, direction_t::W);
        const auto u_right =
            domain.template get_neighbor_value<u_t>(morton_id, direction_t::E);
        const auto v_value = domain.tree().get<v_t>(morton_id);
        const auto v_top =
            domain.template get_neighbor_value<v_t>(morton_id, direction_t::N);
        const auto v_bottom =
            domain.template get_neighbor_value<v_t>(morton_id, direction_t::S);

        const double duT_dx =
            (1.0 / domain.dx(morton_id)) *
                ((u_value + u_right) * 0.25 * (t_value + t_right) -
                 (u_left + u_value) * 0.25 * (t_left + t_value)) +
            (_gamma / domain.dx(morton_id)) *
                (std::abs(u_value + u_right) * 0.25 * (t_value - t_right) -
                 std::abs(u_left + u_value) * 0.25 * (t_left - t_value));

        const double dvT_dy =
            (1.0 / domain.dy(morton_id)) *
                ((v_value + v_top) * 0.25 * (t_value + t_top) -
                 (v_bottom + v_value) * 0.25 * (t_bottom + t_value)) +
            (_gamma / domain.dy(morton_id)) *
                (std::abs(v_value + v_top) * 0.25 * (t_value - t_top) -
                 std::abs(v_bottom + v_value) * 0.25 * (t_bottom - t_value));

        return duT_dx + dvT_dy;
    }

    template <typename ValueType>
    static double laplacian(sim_domain& domain, index_t morton_id)
    {
        const auto value = domain.tree().get<ValueType>(morton_id);
        const auto left_value = domain.template get_neighbor_value<ValueType>(
            morton_id, direction_t::W
        );
        const auto right_value = domain.template get_neighbor_value<ValueType>(
            morton_id, direction_t::E
        );
        const auto top_value = domain.template get_neighbor_value<ValueType>(
            morton_id, direction_t::N
        );
        const auto bottom_value = domain.template get_neighbor_value<ValueType>(
            morton_id, direction_t::S
        );

        const double fd_x = (right_value - 2.0 * value + left_value) /
                            (utils::pow2(domain.dx(morton_id)));
        const double fd_y = (top_value - 2.0 * value + bottom_value) /
                            (utils::pow2(domain.dy(morton_id)));
        return fd_x + fd_y;
    }

    static double sor_helper(sim_domain& domain, index_t morton_id)
    {
        const auto left_p_value =
            domain.template get_neighbor_value<p_t>(morton_id, direction_t::W);
        const auto right_p_value =
            domain.template get_neighbor_value<p_t>(morton_id, direction_t::E);
        const auto top_p_value =
            domain.template get_neighbor_value<p_t>(morton_id, direction_t::N);
        const auto bottom_p_value =
            domain.template get_neighbor_value<p_t>(morton_id, direction_t::S);

        const auto result = (left_p_value + right_p_value) /
                                (utils::pow2(domain.dx(morton_id))) +
                            (top_p_value + bottom_p_value) /
                                (utils::pow2(domain.dy(morton_id)));
        return result;
    }

  private:
    inline static double _gamma;
};
