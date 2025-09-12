#include "ndtree/print_tree_a.hpp"
#include "ndtree/ndtree.hpp"
#include "ndtree/patch.hpp"
#include "morton/morton_id.hpp"

#include "solver/amr_solver.hpp"
#include "solver/cell_types.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <cstdio>
#include <functional>
#include <filesystem>


// Define the types needed for the amr_solver template
using Patch_Type = PatchIndexer<4,4>;
using index_t = amr::ndt::morton::morton_id<7u, 2u>;
using tree_t = amr::ndt::tree::ndtree<amr::cell::EulerCell, index_t, Patch_Type>;

int main() {
    std::cout << "Hello AMR world\n";

    // Ensure output directory exists
    std::filesystem::create_directory("output");

    double tmax = 0.8; // Example tmax, adjust as needed
    int output_interval = 10;
    const std::string output_prefix = "solver_integration_test";

    // Create the VTK printer instance
    ndt::print::vtk_print vtk_printer(output_prefix);

    // Instantiate the AMR solver.
    amr_solver<tree_t> solver(100);

    // Define the initial condition function
    auto quadrantsIC = [](double x, double y) -> std::vector<double> {
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
    };
    solver.initialize(quadrantsIC);

    // --- Main Simulation Loop ---
    double t = 0.0;
    int step = 0;

    char filename[256];
    snprintf(filename, sizeof(filename), "output/%s_%06d.vtk", output_prefix.c_str(), step);
    vtk_printer.print(solver.get_tree(), filename);

    std::cout << "Starting AMR simulation..." << std::endl;

    while (t < tmax) {
        double dt = 0.01; // Using a fixed dt for now  TODO: compute for stability

        solver.time_step(dt);
        
        solver.get_tree().reconstruct_tree(
            [](const auto& node_id) {
                // Placeholder for a real refinement criterion
                // TODO: implement shock based or similar
                if (node_id.level() < 4) {
                    return tree_t::refine_status_t::Refine;
                }
                return tree_t::refine_status_t::Stable;
            }
        );

        t += dt;
        step++;
        
        if (step % output_interval == 0) {
            snprintf(filename, sizeof(filename), "output/%s_%06d.vtk", output_prefix.c_str(), step);
            vtk_printer.print(solver.get_tree(), filename);
            std::cout << "Step: " << step << ", Time: " << t << ", Tree Size: " << solver.get_tree_size() << std::endl;
        }

        if (step > 100) {
            std::cout << "Breaking after 100 steps for safety..." << std::endl;
            break;
        }
    }

    snprintf(filename, sizeof(filename), "output/%s_%06d.vtk", output_prefix.c_str(), step);
    vtk_printer.print(solver.get_tree(), filename);

    std::cout << "Simulation completed. Files in output/ directory." << std::endl;
    
    return 0;
}