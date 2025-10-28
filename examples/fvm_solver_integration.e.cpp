#include "containers/static_layout.hpp"
#include "containers/static_shape.hpp"
#include "containers/static_vector.hpp"
#include "ndtree/print_tree_a.hpp"
#include "ndtree/ndhierarchy.hpp"
#include "ndtree/ndtree.hpp"
#include "ndtree/patch_layout.hpp"
#include "morton/morton_id.hpp"

#include "solver/amr_solver.hpp"
#include "solver/cell_types.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <cstdio>
#include <functional>
#include <filesystem>
#include <cmath>


int main() {
    std::cout << "Hello AMR world\n";
    constexpr std::size_t N    = 16;
    constexpr std::size_t M    = 16;
    constexpr std::size_t Halo = 1;

    using shape_t         = amr::containers::static_shape<N, M>;
    using layout_t        = amr::containers::static_layout<shape_t>;
    //using index_t         = typename layout_t::index_t;

    using patch_index_t  = amr::ndt::morton::morton_id<9u, 2u>;
    using patch_layout_t = amr::ndt::patches::patch_layout<layout_t, Halo>;
    using tree_t         = amr::ndt::tree::ndtree<amr::cell::EulerCell2D, patch_index_t, patch_layout_t>;


    ndt::print::example_patch_print<Halo, M, N> printer("euler_tree");


    double tmax = 90000000; // Example tmax, adjust as needed
    const std::string output_prefix = "solver_integration_test";


    // Instantiate the AMR solver.
    amr_solver<tree_t, 2> solver(100000); // Provide initial capacity for tree

    auto acousticWaveCriterion = [&](const patch_index_t& idx) {
        // Define Thresholds and Limits
        constexpr double REFINE_RHO_THRESHOLD = 2.5; // Refine if Rho > 1.01 (1% above background)
        constexpr double COARSEN_RHO_THRESHOLD = 2; // Coarsen if Rho < 1.005
        constexpr int MAX_LEVEL = 4;
        constexpr int MIN_LEVEL = 0;
        
        int level = idx.level();

        // Compute Jump (using the undivided second difference of density, Rho)
        // ----> first use only placeholder with abs value
        auto rho_patch = solver.get_tree().template get_patch<amr::cell::Rho>(idx);
        double max_rho_value = 0;
        
        for (std::size_t linear_idx = 0; linear_idx != patch_layout_t::flat_size(); ++linear_idx) {
            if (rho_patch[linear_idx] > max_rho_value)
            {
                max_rho_value = rho_patch[linear_idx];
            }
        }

        
        // Hard check for refinement limit
        if (level < MAX_LEVEL && max_rho_value > REFINE_RHO_THRESHOLD) {
            return tree_t::refine_status_t::Refine;
        } 
        
        // Hard check for coarsening limit (only coarsen patches that are already refined)
        if (level > MIN_LEVEL && max_rho_value < COARSEN_RHO_THRESHOLD) {
            return tree_t::refine_status_t::Coarsen;
        }
        
        return tree_t::refine_status_t::Stable;
    };

    // Parameters for the Acoustic Pulse
    constexpr double RHO_BG = 1.0;
    constexpr double P_BG = 1.0;
    constexpr double AMPLITUDE = 5;
    constexpr double PULSE_WIDTH_SQ = 0.01; // sigma^2
    constexpr double CENTER_X = 0.5;
    constexpr double CENTER_Y = 0.5;

    // The initial condition function (auto IC = [](){})
    auto acousticPulseIC = [](double x, double y) -> std::vector<double> {
        std::vector<double> prim(4);
        
        // Calculate distance squared from the center
        double dx = x - CENTER_X;
        double dy = y - CENTER_Y;
        double r_sq = dx * dx + dy * dy;

        // Calculate the density/pressure perturbation
        double perturbation = AMPLITUDE * std::exp(-r_sq / PULSE_WIDTH_SQ);

        // Set primitive variables: [rho, u, v, p]
        prim[0] = RHO_BG + perturbation;  // Density (rho)
        prim[1] = 0.0;                    // X-velocity (u)
        prim[2] = 0.0;                    // Y-velocity (v)
        prim[3] = P_BG + perturbation;    // Pressure (p)
        
        return prim;
    };

    std::cout << "Initializing solver..." << std::endl;
    solver.initialize(acousticPulseIC);

    // CHECK INITIAL CONDITIONS
    std::cout << "\n=== Checking Initial Conditions ===" << std::endl;
    for (size_t patch_idx = 0; patch_idx < std::min(size_t(3), solver.get_tree().size()); ++patch_idx) {
        auto& rho_patch = solver.get_tree().template get_patch<amr::cell::Rho>(patch_idx);
        auto& rhou_patch = solver.get_tree().template get_patch<amr::cell::Rhou>(patch_idx);
        auto& rhov_patch = solver.get_tree().template get_patch<amr::cell::Rhov>(patch_idx);
        auto& e_patch = solver.get_tree().template get_patch<amr::cell::E2D>(patch_idx);
        
        std::cout << "Patch " << patch_idx << " first 3 cells:" << std::endl;
        for (size_t i = 0; i < 3; ++i) {
            std::cout << "  Cell " << i << ": "
                      << "rho=" << rho_patch[i] 
                      << ", rhou=" << rhou_patch[i]
                      << ", rhov=" << rhov_patch[i]
                      << ", E=" << e_patch[i] << std::endl;
        }
    }

    std::cout << "\nWriting initial VTK..." << std::endl;
    printer.print(solver.get_tree(), "_iteration_0.vtk");

    // Main Simulation Loop
    double t = 0.0;
    int step = 1;

    std::cout << "\nStarting AMR simulation..." << std::endl;

    while (t < tmax) {
        // std::cout << "\n=== Step " << step << ", t=" << t << " ===" << std::endl;
        
        // double dt = 300; // Using a fixed dt for now
        double dt = solver.compute_time_step(); // Test adaptive time step
        
        solver.time_step(dt);
        
        // // Check for NaN after timestep
        // bool has_nan = false;
        // for (size_t patch_idx = 0; patch_idx < solver.get_tree().size(); ++patch_idx) {
        //     auto& rho_patch = solver.get_tree().template get_patch<amr::cell::Rho>(patch_idx);
        //     for (size_t i = 0; i < 10; ++i) {
        //         if (!std::isfinite(rho_patch[i])) {
        //             std::cout << "NaN detected in SOLVER at patch " << patch_idx << " cell " << i << std::endl;
        //             std::cout << "  Value: " << rho_patch[i] << std::endl;
        //             has_nan = true;
        //             break;
        //         }
        //     }
        //     if (has_nan) break;
        // }
        
        // if (has_nan) {
        //     std::cout << "STOPPING DUE TO NaN IN SOLVER" << std::endl;
        //     return 1;
        // }
        
        // std::cout << "Step " << step << " completed, no NaN in solver" << std::endl;
        
        // solver.get_tree().reconstruct_tree(acousticWaveCriterion);

        std::string file_extension = "_iteration_" + std::to_string(step) + ".vtk";
        printer.print(solver.get_tree(), file_extension);

        t += dt;
        step++;

        if (step > 100) {
            std::cout << "Breaking after 100 steps for testing..." << std::endl;
            break;
        }
    }

    std::cout << "\nSimulation completed. Files in vtk_output/ directory." << std::endl;
    
    return 0;
}