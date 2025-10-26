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
    using tree_t         = amr::ndt::tree::ndtree<amr::cell::EulerCell, patch_index_t, patch_layout_t>;


    ndt::print::example_patch_print<Halo, M, N> printer("euler_tree");


    double tmax = 90000000; // Example tmax, adjust as needed
    const std::string output_prefix = "solver_integration_test";


    // Instantiate the AMR solver.
    amr_solver<tree_t> solver(100000); // Provide initial capacity for tree

    // Define the initial condition function -> 2D Riemann problem, will produce complex interacting waves and shocks
    /*auto quadrantsIC = [](double x, double y) -> std::vector<double> {
        std::vector<double> prim(4);
        // Define regions based on ClawPack
        bool right = (x >= 0.8);
        bool top = (y >= 0.8);
        
        // Set primitive variables based on quadrant
        if (right && top) {         // top-right
            prim[0] = 1.5;                // density
            prim[1] = 0.0;                // u velocity
            prim[2] = 0.0;                // v velocity
            prim[3] = 1.5;                // pressure
        } else if (!right && top) { // top-left
            prim[0] = 0.532258064516129;
            prim[1] = 1.206045378311055;
            prim[2] = 0.0;
            prim[3] = 0.3;
        } else if (!right && !top) { // bottom-left
            prim[0] = 0.137992831541219;
            prim[1] = 1.206045378311055;
            prim[2] = 1.206045378311055;
            prim[3] = 0.029032258064516;
        } else {                    // bottom-right
            prim[0] = 0.532258064516129;
            prim[1] = 0.0;
            prim[2] = 1.206045378311055;
            prim[3] = 0.3;
        }
        return prim;
    };*/

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

    solver.initialize(acousticPulseIC);

    // --- Main Simulation Loop ---
    double t = 0.0;
    int step = 1;

    std::cout << "Starting AMR simulation..." << std::endl;

    printer.print(solver.get_tree(), "_iteration_0.vtk");

    while (t < tmax) {
        double dt = 300; // Using a fixed dt for now

        solver.time_step(dt);
        
        solver.get_tree().reconstruct_tree(acousticWaveCriterion);

        std::string file_extension = "_iteration_" + std::to_string(step) + ".vtk";
        printer.print(solver.get_tree(), file_extension);

        t += dt;
        step++;

        if (step > 10) {
            std::cout << "Breaking after 100 steps for safety..." << std::endl;
            break;
        }
    }

    std::cout << "Simulation completed. Files in output/ directory." << std::endl;
    
    return 0;
}