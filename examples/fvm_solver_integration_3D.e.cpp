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
    constexpr std::size_t O    = 16;
    constexpr std::size_t Halo = 1;

    using shape_t         = amr::containers::static_shape<N, M, O>;
    using layout_t        = amr::containers::static_layout<shape_t>;
    //using index_t         = typename layout_t::index_t;

    using patch_index_t  = amr::ndt::morton::morton_id<9u, 3u>;
    using patch_layout_t = amr::ndt::patches::patch_layout<layout_t, Halo>;
    using tree_t         = amr::ndt::tree::ndtree<amr::cell::EulerCell3D, patch_index_t, patch_layout_t>;

    
    ndt::print::example_patch_print<Halo, N, M, O> printer("euler_tree"); // NOTE: NEED TO IMPLEMENT PRINT 3D FUNCTIONALITY (?)


    double tmax = 90000000; // Example tmax, adjust as needed
    const std::string output_prefix = "solver_integration_test";

    // Instantiate the AMR solver.
    amr_solver<tree_t, 3> solver(100000); // Provide initial capacity for tree

    // Refinement criterion for 3D acoustic wave
    auto acousticWaveCriterion3D = [&](const patch_index_t& idx) {
        // Define Thresholds and Limits
        constexpr double REFINE_RHO_THRESHOLD = 2.5;  // Refine if Rho > 2.5
        constexpr double COARSEN_RHO_THRESHOLD = 2.0; // Coarsen if Rho < 2.0
        constexpr int MAX_LEVEL = 4;
        constexpr int MIN_LEVEL = 0;
        
        int level = idx.level();

        // Compute Jump (using the undivided second difference of density, Rho)
        // ----> first use only placeholder with abs value
        auto rho_patch = solver.get_tree().template get_patch<amr::cell:: Rho>(idx);
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

    // Parameters for the 3D Acoustic Pulse
    constexpr double RHO_BG = 1.0;
    constexpr double P_BG = 1.0;
    constexpr double AMPLITUDE = 5.0;
    constexpr double PULSE_WIDTH_SQ = 0.01; // sigma^2
    constexpr double CENTER_X = 0.5;
    constexpr double CENTER_Y = 0.5;
    constexpr double CENTER_Z = 0.5;

    // The initial condition function (auto IC = [](){})
    auto acousticPulseIC_3D = [](double x, double y, double z) -> std::vector<double> {
        std::vector<double> prim(5);
        
        // Calculate distance squared from the center
        double dx = x - CENTER_X;
        double dy = y - CENTER_Y;
        double dz = z - CENTER_Z;
        double r_sq = dx * dx + dy * dy + dz * dz;

        // Calculate the density/pressure perturbation
        double perturbation = AMPLITUDE * std::exp(-r_sq / PULSE_WIDTH_SQ);

        // Set primitive variables: [rho, u, v, w, p]
        prim[0] = RHO_BG + perturbation;  // Density (rho)
        prim[1] = 0.0;                     // X-velocity (u)
        prim[2] = 0.0;                     // Y-velocity (v)
        prim[3] = 0.0;                     // Z-velocity (w)
        prim[4] = P_BG + perturbation;     // Pressure (p)
        
        return prim;
    };

    solver.initialize(acousticPulseIC_3D);

    // --- Main Simulation Loop ---
    double t = 0.0;
    int step = 1;

    std::cout << "Starting AMR simulation..." << std::endl;

    printer.print(solver.get_tree(), "_iteration_0.vtk");

    while (t < tmax) {
        double dt = 300; // Using a fixed dt for now
        // double dt = solver.compute_time_step(); // Test adaptive time step

        solver.time_step(dt);
        
        // solver.get_tree().reconstruct_tree(acousticWaveCriterion3D);

        std::string file_extension = "_iteration_" + std::to_string(step) + ".vtk";
        printer.print(solver.get_tree(), file_extension);

        t += dt;
        step++;

        if (step > 100) {
            std::cout << "Breaking after 100 steps for safety..." << std::endl;
            break;
        }
    }

    std::cout << "Simulation completed. Files in output/ directory." << std::endl;
    
    return 0;
}