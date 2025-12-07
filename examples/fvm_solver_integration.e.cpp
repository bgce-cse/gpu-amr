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
    constexpr std::size_t N    = 10;
    constexpr std::size_t M    = 10;
    constexpr std::size_t Halo = 2;

    using shape_t         = amr::containers::static_shape<N, M>;
    using layout_t        = amr::containers::static_layout<shape_t>;
    //using index_t         = typename layout_t::index_t;

    using patch_index_t  = amr::ndt::morton::morton_id<7u, 2u>;
    using patch_layout_t = amr::ndt::patches::patch_layout<layout_t, Halo>;
    using tree_t         = amr::ndt::tree::ndtree<amr::cell::EulerCell2D, patch_index_t, patch_layout_t>;


    amr::ndt::print::example_halo_patch_print<Halo, M, N> printer("euler_halo");
    amr::ndt::print::example_patch_print<Halo, M, N> printer2("euler_tree");


    double tmax = 500; // Example tmax, adjust as needed
    double print_frequency = 10.0; // Print every 10 seconds
    const std::string output_prefix = "solver_integration_test_refine";


    // Instantiate the AMR solver.
    amr_solver<tree_t, 2> solver(1000000); // Provide initial capacity for tree


    auto refineAll = [&]([[maybe_unused]] const patch_index_t& idx) {
    return tree_t::refine_status_t::Refine;
    };

    auto acousticWaveCriterion = [&](const patch_index_t& idx) {
        // Define Thresholds and Limits
        constexpr double REFINE_RHO_THRESHOLD = 0.55;
        constexpr double COARSEN_RHO_THRESHOLD = 0.47;
        constexpr int MAX_LEVEL = 5;
        constexpr int MIN_LEVEL = 1;
        
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
    constexpr double RHO_BG = 0.5;
    constexpr double P_BG = 1.0;
    constexpr double AMPLITUDE = 10.0;
    constexpr double PULSE_WIDTH_SQ = 0.01; // sigma^2
    constexpr double CENTER_X = 0.5;
    constexpr double CENTER_Y = 0.5;

    // The initial condition function (auto IC = [](){})
    auto acousticPulseIC = [](double x, double y) -> amr::containers::static_vector<double, 4> {
        amr::containers::static_vector<double, 4> prim;
        
        // Calculate distance squared from the center
        double dx = x - CENTER_X;
        double dy = y - CENTER_Y;
        double r_sq = dx * dx + dy * dy;

        // Calculate the density/pressure perturbation
        double perturbation = AMPLITUDE * std::exp(-r_sq / PULSE_WIDTH_SQ);

        // Set primitive variables: [rho, u, v, p]
        prim[0] = RHO_BG + (perturbation*0.2);  // Density (rho)
        prim[1] = 0.0;                    // X-velocity (u)
        prim[2] = 0.0;                    // Y-velocity (v)
        prim[3] = P_BG + perturbation;    // Pressure (p)
        
        return prim;
    };

    std::cout << "Initializing solver..." << std::endl;
    solver.get_tree().reconstruct_tree(refineAll);
    solver.get_tree().halo_exchange_update();
    solver.initialize(acousticPulseIC);
    solver.get_tree().halo_exchange_update();

    // Print initial state
    printer.print(solver.get_tree(), "_iteration_0.vtk");

    // Main Simulation Loop
    double t = 0.0;
    double next_print_time = print_frequency;
    int step = 1;
    int output_counter = 1;

    std::cout << "\nStarting AMR simulation..." << std::endl;

    while (t < tmax) {
        double dt = solver.compute_time_step();

        std::cout << "Step " << step << ", t=" << t << ", dt=" << dt << std::endl;
        
        solver.time_step(dt);
        
        if (step % 5 == 0)
        {
            solver.get_tree().reconstruct_tree(acousticWaveCriterion);
            solver.get_tree().halo_exchange_update();
        }

        t += dt;
        
        // Print only when we've passed the next print time
        if (t >= next_print_time) {
            std::string file_extension = "_iteration_" + std::to_string(output_counter) + ".vtk";
            printer.print(solver.get_tree(), file_extension);
            printer2.print(solver.get_tree(), file_extension);
            next_print_time += print_frequency;
            output_counter++;
        }

        step++;

    }

    std::cout << "\nSimulation completed. Files in vtk_output/ directory." << std::endl;
    
    return 0;
}