#include "containers/static_layout.hpp"
#include "containers/static_shape.hpp"
#include "containers/static_vector.hpp"
#include "morton/morton_id.hpp"
#include "ndtree/intergrid_operator.hpp"
#include "ndtree/ndhierarchy.hpp"
#include "ndtree/ndtree.hpp"
#include "ndtree/patch_layout.hpp"
#include "solver/EulerPhysics.hpp"
#include "solver/amr_solver.hpp"
#include "solver/cell_types.hpp"
#include "solver/error_norms.hpp"
#include "solver/physics_system.hpp"
#include "utility/logging.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>

#define VTK_PRINT (false)

int main()
{
    std::cout << "Hello AMR world\n";

    constexpr std::size_t N    = 10;
    constexpr std::size_t M    = 10;
    constexpr std::size_t Halo = 2;

    constexpr double physics_x = 1.0;
    constexpr double physics_y = 1.0;
    constexpr double t_output  = 5.0;

    constexpr std::array<double, 2> physics_lengths = { physics_x, physics_y };

    using shape_t  = amr::containers::static_shape<N, M>;
    using layout_t = amr::containers::static_layout<shape_t>;

    using patch_index_t  = amr::ndt::morton::morton_id<7u, 2u>;
    using patch_layout_t = amr::ndt::patches::patch_layout<layout_t, Halo>;
    using intergrid_operator_t =
        amr::ndt::intergrid_operator::linear_interpolator<patch_layout_t>;

    using tree_t = amr::ndt::tree::ndtree<
        amr::cell::EulerCell2D,
        patch_index_t,
        patch_layout_t,
        intergrid_operator_t>;

    using physics_t =
        amr::ndt::solver::physics_system<patch_index_t, patch_layout_t, physics_lengths>;

#if VTK_PRINT
    amr::ndt::print::vtk_print<physics_t> printer("euler_print");
#endif

    constexpr int initial_refinement = 3;

    amr_solver<tree_t, physics_t, EulerPhysics2D, 2> solver(1000000, 1.4, 0.3);

    auto refineAll = [&]([[maybe_unused]] const patch_index_t& idx)
    {
        return tree_t::refine_status_t::Refine;
    };

    // Keep this if you later want to test AMR regridding with the manufactured case.
    auto acousticWaveCriterion = [&](const patch_index_t& idx)
    {
        constexpr double REFINE_RHO_THRESHOLD  = 0.53;
        constexpr double COARSEN_RHO_THRESHOLD = 0.49;
        constexpr int    MAX_LEVEL             = 6;
        constexpr int    MIN_LEVEL             = 1;

        const int level = idx.level();

        const auto rho_patch =
            solver.get_tree().template get_patch<amr::cell::Rho>(idx);

        double max_rho_value = 0.0;
        for (std::size_t linear_idx = 0; linear_idx != patch_layout_t::flat_size();
             ++linear_idx)
        {
            max_rho_value = std::max(rho_patch[linear_idx], max_rho_value);
        }

        if (level < MAX_LEVEL && max_rho_value > REFINE_RHO_THRESHOLD)
        {
            return tree_t::refine_status_t::Refine;
        }
        if (level > MIN_LEVEL && max_rho_value < COARSEN_RHO_THRESHOLD)
        {
            return tree_t::refine_status_t::Coarsen;
        }
        return tree_t::refine_status_t::Stable;
    };

    // Manufactured/advection-style smooth test
    constexpr double EPS = 0.1;
    constexpr double C   = 1.0;

    auto manufacturedIC =
        [=](auto const& coords) -> amr::containers::static_vector<double, 4>
    {
        const double x = coords[0];
        const double y = coords[1];

        amr::containers::static_vector<double, 4> prim;

        const double rho =
            1.0 + EPS * std::sin(2.0 * M_PI * (x / physics_x + y / physics_y));
        const double u = C;
        const double v = C;
        const double p = 1.0;

        prim[0] = rho;
        prim[1] = u;
        prim[2] = v;
        prim[3] = p;

        return prim;
    };

    std::cout << "Initializing solver...\n";
    for (int i = 0; i < initial_refinement; ++i)
    {
        solver.get_tree().reconstruct_tree(refineAll);
        solver.get_tree().halo_exchange_update();
    }

    solver.initialize(manufacturedIC);
    solver.get_tree().halo_exchange_update();

#if VTK_PRINT
    printer.print(solver.get_tree(), "_iteration_0.vtk");
#endif

    double t    = 0.0;
    int    step = 1;

    std::size_t cell_update_count = 0;
    const auto  start             = std::chrono::steady_clock::now();

    std::cout << "\nStarting AMR simulation...\n";

    while (t < t_output)
    {
        double dt = solver.compute_time_step();
        if (t + dt > t_output)
        {
            dt = t_output - t;
        }

        DEFAULT_SOURCE_LOG_PROGRESS("Step: {},\tt: {:.5f},\tdt: {:.5f}", step, t, dt);

        solver.time_step(dt);
        cell_update_count +=
            solver.get_tree().size() * patch_layout_t::data_layout_t::flat_size();

        solver.get_tree().halo_exchange_update();

        // For a clean analytical test, leave regridding off at first.
        // If you later want AMR-on verification, uncomment this block.
        /*
        if (step % 5 == 0)
        {
            solver.get_tree().reconstruct_tree(acousticWaveCriterion);
            solver.get_tree().halo_exchange_update();
        }
        */

        t += dt;
        ++step;
    }

    const auto                          end      = std::chrono::steady_clock::now();
    const std::chrono::duration<double> duration = end - start;

    std::cout << "Final t = " << t << "\n";

    // Exact translated solution
    auto manufacturedIC_xy =
        [=](double x, double y) -> amr::containers::static_vector<double, 4>
    {
        amr::containers::static_vector<double, 4> prim;

        const double rho =
            1.0 + EPS * std::sin(2.0 * M_PI * (x / physics_x + y / physics_y));
        const double u = C;
        const double v = C;
        const double p = 1.0;

        prim[0] = rho;
        prim[1] = u;
        prim[2] = v;
        prim[3] = p;

        return prim;
    };

    auto exact_solution_prim = make_exact_advection_solution_2d(
        manufacturedIC_xy,
        C, C,
        0.0, physics_x,
        0.0, physics_y,
        true
    );

    auto exact_solution_cons = [&](double x, double y, double t_now)
    {
        const auto prim = exact_solution_prim(x, y, t_now);

        amr::containers::static_vector<double, 4> cons;
        EulerPhysics2D::primitiveToConservative(prim, cons, 1.4);
        return cons;
    };

    const auto err = compute_error_norms_euler_2d<tree_t, physics_t>(
        solver.get_tree(),
        t,
        exact_solution_cons
    );

    print_error_norms<4>(
        err,
        { "rho", "rhou", "rhov", "E" }
    );

    std::printf(
        "N=%zu  L1(rho)=%.8e  L2(rho)=%.8e  Linf(rho)=%.8e\n",
        N, err.l1[0], err.l2[0], err.linf[0]
    );

    std::cout << "Updated cells: " << cell_update_count << '\n';
    std::cout << "Duration: " << duration.count() << '\n';
    std::cout << "Updates per second: "
              << static_cast<double>(cell_update_count) / duration.count() << '\n';

    std::cout << "\nSimulation completed.\n";
    return 0;
}