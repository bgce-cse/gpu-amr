#include "containers/static_layout.hpp"
#include "containers/static_shape.hpp"
#include "containers/static_vector.hpp"
#include "morton/morton_id.hpp"
#include "ndtree/ndhierarchy.hpp"
#include "ndtree/ndtree.hpp"
#include "ndtree/patch_layout.hpp"
#include "solver/amr_solver.hpp"
#include "solver/cell_types.hpp"
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

int main()
{
    std::cout << "Hello AMR world\n";
    constexpr std::size_t N    = 16;
    constexpr std::size_t M    = 16;
    constexpr std::size_t O    = 16;
    constexpr std::size_t Halo = 1;

    using shape_t  = amr::containers::static_shape<N, M, O>;
    using layout_t = amr::containers::static_layout<shape_t>;

    using patch_index_t  = amr::ndt::morton::morton_id<9u, 3u>;
    using patch_layout_t = amr::ndt::patches::patch_layout<layout_t, Halo>;
    using tree_t =
        amr::ndt::tree::ndtree<amr::cell::EulerCell3D, patch_index_t, patch_layout_t>;

    ndt::print::example_patch_print<Halo, N, M, O> printer("euler_tree_3d");

    double            tmax          = 90000000; // Example tmax, adjust as needed
    const std::string output_prefix = "solver_integration_test_3d";

    // Instantiate the AMR solver.
    amr_solver<tree_t, 3> solver(100000); // Provide initial capacity for tree

    // Refinement criterion for 3D acoustic wave
    auto acousticWaveCriterion3D = [&](const patch_index_t& idx)
    {
        // Define Thresholds and Limits
        constexpr double REFINE_RHO_THRESHOLD =
            2.5; // Refine if Rho > 1.01 (1% above background)
        constexpr double COARSEN_RHO_THRESHOLD = 2.0; // Coarsen if Rho < 1.005
        constexpr int    MAX_LEVEL             = 4;
        constexpr int    MIN_LEVEL             = 0;

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

    // Parameters for the 3D Acoustic Pulse
    constexpr double RHO_BG         = 1.0;
    constexpr double P_BG           = 1.0;
    constexpr double AMPLITUDE      = 5.0;
    constexpr double PULSE_WIDTH_SQ = 0.01;
    constexpr double CENTER_X       = 0.5;
    constexpr double CENTER_Y       = 0.5;
    constexpr double CENTER_Z       = 0.5;

    // Initial condition function for 3D
    auto acousticPulseIC_3D =
        [](double x, double y, double z) -> amr::containers::static_vector<double, 5>
    {
        amr::containers::static_vector<double, 5> prim;

        // Calculate distance squared from the center
        double dx   = x - CENTER_X;
        double dy   = y - CENTER_Y;
        double dz   = z - CENTER_Z;
        double r_sq = dx * dx + dy * dy + dz * dz;

        // Calculate the density/pressure perturbation
        double perturbation = AMPLITUDE * std::exp(-r_sq / PULSE_WIDTH_SQ);

        // Set primitive variables: [rho, u, v, w, p]
        prim[0] = RHO_BG + perturbation; // Density (rho)
        prim[1] = 0.0;                   // X-velocity (u)
        prim[2] = 0.0;                   // Y-velocity (v)
        prim[3] = 0.0;                   // Z-velocity (w)
        prim[4] = P_BG + perturbation;   // Pressure (p)

        return prim;
    };

    std::cout << "Initializing solver..." << std::endl;
    solver.initialize(acousticPulseIC_3D);

    // Main Simulation Loop
    double t    = 0.0;
    int    step = 1;

    std::cout << "Starting AMR simulation..." << std::endl;

    printer.print(solver.get_tree(), "_iteration_0.vtk");

    while (t < tmax)
    {
        // double dt = 300; // Using a fixed dt for now
        double dt = solver.compute_time_step(); // Adaptive time step

        solver.time_step(dt);

        // // start: DEBUG for NAN values
        // bool has_nan = false;
        // double max_rho = -1e10, min_rho = 1e10;
        // double max_rhou = -1e10, min_rhou = 1e10;
        // double max_rhov = -1e10, min_rhov = 1e10;
        // double max_rhow = -1e10, min_rhow = 1e10;
        // double max_e = -1e10, min_e = 1e10;

        // size_t nan_patch = 0;
        // size_t nan_cell = 0;
        // std::string nan_variable = "";

        // constexpr size_t patch_flat_size = patch_layout_t::flat_size();

        // for (size_t patch_idx = 0; patch_idx < solver.get_tree().size(); ++patch_idx) {
        //     auto& rho_patch = solver.get_tree().template
        //     get_patch<amr::cell::Rho>(patch_idx); auto& rhou_patch =
        //     solver.get_tree().template get_patch<amr::cell::Rhou>(patch_idx); auto&
        //     rhov_patch = solver.get_tree().template
        //     get_patch<amr::cell::Rhov>(patch_idx); auto& rhow_patch =
        //     solver.get_tree().template get_patch<amr::cell::Rhow>(patch_idx); auto&
        //     e_patch = solver.get_tree().template get_patch<amr::cell::E3D>(patch_idx);

        //     for (size_t i = 0; i < patch_flat_size; ++i) {
        //         // Check Rho
        //         if (std::isfinite(rho_patch[i])) {
        //             max_rho = std::max(max_rho, rho_patch[i]);
        //             min_rho = std::min(min_rho, rho_patch[i]);
        //         } else if (!has_nan) {
        //             has_nan = true;
        //             nan_variable = "Rho";
        //             nan_patch = patch_idx;
        //             nan_cell = i;
        //         }

        //         // Check Rhou
        //         if (std::isfinite(rhou_patch[i])) {
        //             max_rhou = std::max(max_rhou, rhou_patch[i]);
        //             min_rhou = std::min(min_rhou, rhou_patch[i]);
        //         } else if (!has_nan) {
        //             has_nan = true;
        //             nan_variable = "Rhou";
        //             nan_patch = patch_idx;
        //             nan_cell = i;
        //         }

        //         // Check Rhov
        //         if (std::isfinite(rhov_patch[i])) {
        //             max_rhov = std::max(max_rhov, rhov_patch[i]);
        //             min_rhov = std::min(min_rhov, rhov_patch[i]);
        //         } else if (!has_nan) {
        //             has_nan = true;
        //             nan_variable = "Rhov";
        //             nan_patch = patch_idx;
        //             nan_cell = i;
        //         }

        //         // Check Rhow
        //         if (std::isfinite(rhow_patch[i])) {
        //             max_rhow = std::max(max_rhow, rhow_patch[i]);
        //             min_rhow = std::min(min_rhow, rhow_patch[i]);
        //         } else if (!has_nan) {
        //             has_nan = true;
        //             nan_variable = "Rhow";
        //             nan_patch = patch_idx;
        //             nan_cell = i;
        //         }

        //         // Check E
        //         if (std::isfinite(e_patch[i])) {
        //             max_e = std::max(max_e, e_patch[i]);
        //             min_e = std::min(min_e, e_patch[i]);
        //         } else if (!has_nan) {
        //             has_nan = true;
        //             nan_variable = "E";
        //             nan_patch = patch_idx;
        //             nan_cell = i;
        //         }
        //     }
        //     if (has_nan) break;
        // }

        // // Print stats every step
        // if (has_nan) {
        //     std::cout << "\nNan detected in solver\n";
        //     std::cout << "Step: " << step << "\n";
        //     std::cout << "Time: " << t << "\n";
        //     std::cout << "Variable: " << nan_variable << "\n";
        //     std::cout << "Patch: " << nan_patch << "\n";
        //     std::cout << "Cell: " << nan_cell << "\n";
        //     return 1;
        // }

        // std::cout << "Step " << step << ": ";
        // std::cout << "Rho[" << min_rho << "," << max_rho << "] ";
        // std::cout << "Rhou[" << min_rhou << "," << max_rhou << "] ";
        // std::cout << "Rhov[" << min_rhov << "," << max_rhov << "] ";
        // std::cout << "Rhow[" << min_rhow << "," << max_rhow << "] ";
        // std::cout << "E[" << min_e << "," << max_e << "] ";
        // std::cout << "- OK" << std::endl;
        // // end: DEBUG for NAN values

        // solver.get_tree().reconstruct_tree(acousticWaveCriterion3D);

        std::string file_extension = "_iteration_" + std::to_string(step) + ".vtk";
        printer.print(solver.get_tree(), file_extension);

        t += dt;
        step++;

        if (step > 100)
        {
            std::cout << "Breaking after 100 steps for testing..." << std::endl;
            break;
        }
    }

    std::cout << "Simulation completed. Files in vtk_output/ directory." << std::endl;

    return 0;
}
