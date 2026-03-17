#pragma once

#include "solver/cell_types.hpp"
#include "ndtree/patch_utils.hpp"
#include <array>
#include <cmath>
#include <cstdio>
#include <string_view>
#include <algorithm>

template <std::size_t NVAR>
struct ErrorNorms
{
    std::array<double, NVAR> l1{};
    std::array<double, NVAR> l2{};
    std::array<double, NVAR> linf{};
};

template <std::size_t NVAR>
inline void print_error_norms(
    const ErrorNorms<NVAR>& err,
    const std::array<std::string_view, NVAR>& names
)
{
    std::printf("Errors for each variable\n");
    std::printf("Var\tL1\t\t\tL2\t\t\tLinf\n");
    for (std::size_t k = 0; k < NVAR; ++k)
    {
        std::printf(
            "%s\t%15.8e\t%15.8e\t%15.8e\n",
            names[k].data(),
            err.l1[k],
            err.l2[k],
            err.linf[k]
        );
    }
}

inline void print_rho_error_line(
    std::size_t N,
    const ErrorNorms<4>& err
)
{
    std::printf(
        "N=%zu L1(rho)=%.8e L2(rho)=%.8e Linf(rho)=%.8e\n",
        N, err.l1[0], err.l2[0], err.linf[0]
    );
}

template <typename TreeT, typename PhysicsT, typename ExactFunc>
auto compute_error_norms_euler_2d(
    const TreeT& tree,
    double t,
    ExactFunc exact_solution
) -> ErrorNorms<4>
{
    constexpr std::size_t NVAR = 4;
    using patch_layout_t = typename TreeT::patch_layout_t;

    ErrorNorms<NVAR> err{};

    for (std::size_t patch_idx = 0; patch_idx < tree.size(); ++patch_idx)
    {
        const auto patch_id = tree.get_node_index_at(patch_idx);

        const auto& rho_patch  = tree.template get_patch<amr::cell::Rho>(patch_idx);
        const auto& rhou_patch = tree.template get_patch<amr::cell::Rhou>(patch_idx);
        const auto& rhov_patch = tree.template get_patch<amr::cell::Rhov>(patch_idx);
        const auto& e_patch    = tree.template get_patch<amr::cell::E2D>(patch_idx);

        const auto cell_size = PhysicsT::cell_sizes(patch_id);
        const double dx = cell_size[0];
        const double dy = cell_size[1];
        const double area = dx * dy;

        for (std::size_t linear_idx = 0; linear_idx < patch_layout_t::flat_size(); ++linear_idx)
        {
            if (amr::ndt::utils::patches::is_halo_cell<patch_layout_t>(linear_idx))
            {
                continue;
            }

            const auto cell_corner = PhysicsT::cell_coord(patch_id, linear_idx);
            const double x = cell_corner[0] + 0.5 * dx;
            const double y = cell_corner[1] + 0.5 * dy;

            const auto exact = exact_solution(x, y, t);

            const std::array<double, NVAR> num = {
                rho_patch[linear_idx],
                rhou_patch[linear_idx],
                rhov_patch[linear_idx],
                e_patch[linear_idx]
            };

            for (int k = 0; k < static_cast<int>(NVAR); ++k)
            {
                const std::size_t ku = static_cast<std::size_t>(k);
                const double diff = std::abs(num[ku] - exact[k]);
                err.l1[ku]   += area * diff;
                err.l2[ku]   += area * diff * diff;
                err.linf[ku]  = std::max(err.linf[ku], diff);
            }
        }
    }

    for (std::size_t k = 0; k < NVAR; ++k)
    {
        err.l2[k] = std::sqrt(err.l2[k]);
    }

    return err;
}

template <typename InitFunc>
auto make_exact_advection_solution_2d(
    InitFunc init_func,
    double vx,
    double vy,
    double x_min,
    double x_max,
    double y_min,
    double y_max,
    bool periodic = true
)
{
    auto wrap = [](double a, double lo, double hi) -> double
    {
        const double L = hi - lo;
        while (a < lo) a += L;
        while (a >= hi) a -= L;
        return a;
    };

    return [=](double x, double y, double t)
    {
        double x0 = x - vx * t;
        double y0 = y - vy * t;

        if (periodic)
        {
            x0 = wrap(x0, x_min, x_max);
            y0 = wrap(y0, y_min, y_max);
        }

        return init_func(x0, y0);
    };
}