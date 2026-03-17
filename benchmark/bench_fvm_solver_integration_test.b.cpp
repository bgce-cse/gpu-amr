#include "containers/static_layout.hpp"
#include "containers/static_shape.hpp"
#include "containers/static_vector.hpp"
#include "morton/morton_id.hpp"
#include "ndtree/ndhierarchy.hpp"
#include "ndtree/ndtree.hpp"
#include "ndtree/patch_layout.hpp"
#include "solver/amr_solver_MUSCL.hpp"
#include "solver/cell_types.hpp"
#include "solver/error_norms.hpp"
#include "solver/EulerPhysics.hpp"
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <iostream>
#include <string>
#include "utility/logging.hpp"
#include <vector>
#include <fstream>


int main()
{
    std::cout << "Hello AMR world\n";
    constexpr std::size_t N    = 10;
    constexpr std::size_t M    = 20;
    constexpr std::size_t Halo = 2;

    constexpr double physics_x = 1.0;
    constexpr double physics_y = 1.0;
    double t_output = 5;


    constexpr std::array<double, 2> physics_lengths = {physics_x, physics_y};

    using shape_t  = amr::containers::static_shape<N, M>;
    using layout_t = amr::containers::static_layout<shape_t>;
    // using index_t         = typename layout_t::index_t;

    using patch_index_t  = amr::ndt::morton::morton_id<8u, 2u>;
    using patch_layout_t = amr::ndt::patches::patch_layout<layout_t, Halo>;
    using tree_t =
        amr::ndt::tree::ndtree<amr::cell::EulerCell2D, patch_index_t, patch_layout_t>;

    using physics_t = amr::ndt::solver::physics_system<patch_index_t, patch_layout_t, physics_lengths>;

    [[maybe_unused]] double            tmax          = 400;

    [[maybe_unused]] int inital_refinement = 3;

    // Instantiate the AMR solver.
    amr_solver_MUSCL<tree_t,physics_t, 2> solver(1000000); // Provide initial capacity for tree

    [[maybe_unused]] auto refineAll = [&]([[maybe_unused]]
                         const patch_index_t& idx)
    {
        return tree_t::refine_status_t::Refine;
    };

    [[maybe_unused]]
    auto acousticWaveCriterion = [&](const patch_index_t& idx)
    {
        // Define Thresholds and Limits
        constexpr double REFINE_RHO_THRESHOLD  = 0.53;
        constexpr double COARSEN_RHO_THRESHOLD = 0.49;
        constexpr int    MAX_LEVEL             = 6;
        constexpr int    MIN_LEVEL             = 1;

        int level = idx.level();

        // Compute Jump (using the undivided second difference of density, Rho)
        // ----> first use only placeholder with abs value
        auto   rho_patch     = solver.get_tree().template get_patch<amr::cell::Rho>(idx);
        double max_rho_value = 0;

        for (std::size_t linear_idx = 0; linear_idx != patch_layout_t::flat_size();
             ++linear_idx)
        {
            if (rho_patch[linear_idx] > max_rho_value)
            {
                max_rho_value = rho_patch[linear_idx];
            }
        }

        // Hard check for refinement limit
        if (level < MAX_LEVEL && max_rho_value > REFINE_RHO_THRESHOLD)
        {
            return tree_t::refine_status_t::Refine;
        }

        // Hard check for coarsening limit (only coarsen patches that are already refined)
        if (level > MIN_LEVEL && max_rho_value < COARSEN_RHO_THRESHOLD)
        {
            return tree_t::refine_status_t::Coarsen;
        }

        return tree_t::refine_status_t::Stable;
    };

    // Parameters for the Acoustic Pulse
    //constexpr double RHO_BG         = 0.5;
    //constexpr double P_BG           = 1.0;
    //constexpr double AMPLITUDE      = 10.0;
    //constexpr double PULSE_WIDTH_SQ = 0.01; // sigma^2
    //constexpr double CENTER_X       = 0.5;
    //constexpr double CENTER_Y       = 0.5;

    // The initial condition function (auto IC = [](){})
    constexpr double EPS = 0.1;
    constexpr double C   = 1.0;

    auto manufacturedIC = [=](double x, double y)
        -> amr::containers::static_vector<double, 4>
    {
        amr::containers::static_vector<double, 4> prim;

        //double rho = 1.0 + EPS * std::sin(2.0 * M_PI * (x + y));
        double rho = 1.0 + EPS * std::sin(2.0 * M_PI * (x / physics_x + y / physics_y));
        double u   = C;
        double v   = C;
        double p   = 1.0;

        prim[0] = rho;
        prim[1] = u;
        prim[2] = v;
        prim[3] = p;

        return prim;
    };


    solver.initialize(manufacturedIC);
    solver.get_tree().halo_exchange_update();

    // Main Simulation Loop
    double t    = 0.0;
    int    step = 1;

    std::cout << "\nStarting AMR simulation...\n";

    while (t < t_output)
    {
        double dt = solver.compute_time_step();
        if (t + dt > t_output) dt = t_output - t;

        solver.time_step(dt);
        solver.get_tree().halo_exchange_update();

        t += dt;
        step++;
    }
    std::cout << "Final t = " << t << "\n";

    auto exact_solution_prim = make_exact_advection_solution_2d(
        manufacturedIC,
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

    /*
    std::vector<double> rho_snapshot;

    auto& tree = solver.get_tree();

    for (std::size_t patch_idx = 0; patch_idx < tree.size(); ++patch_idx) {
        const auto& rho_patch = tree.template get_patch<amr::cell::Rho>(patch_idx);

        for (std::size_t i = 0; i < patch_layout_t::flat_size(); ++i) {
            if (amr::ndt::utils::patches::is_halo_cell<patch_layout_t>(i)) continue;
            rho_snapshot.push_back(rho_patch[i]);
        }
    }

    std::cout << "\nSimulation completed.\n";
    */

    return 0;
}
