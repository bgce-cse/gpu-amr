#include "solver/AdvectionPhysics.hpp"
#include "solver/amr_solver.hpp"
#include "ndtree/ndtree.hpp"
#include "ndtree/vtk_print.hpp"
#include <cmath>
#include <iostream>

int main() {
    std::cout << "Hello Advection world\n";

    // --- 1. Simulation Parameters ---
    constexpr int DIM = 2;
    constexpr std::size_t N = 16; // 16x16 cells per patch
    constexpr std::size_t Halo = 2;
    constexpr double domain_size = 1.0;
    
    using Physics = AdvectionPhysics<DIM>; // .............
    using Cell = amr::cell::AdvectionCell;

    // --- 2. Boilerplate Types ---
    using shape_t = amr::containers::static_shape<N, N>;
    using layout_t = amr::containers::static_layout<shape_t>;
    using patch_index_t = amr::ndt::morton::morton_id<7u, DIM>;
    using patch_layout_t = amr::ndt::patches::patch_layout<layout_t, Halo>;
    
    using tree_t = amr::ndt::tree::ndtree<Cell, patch_index_t, patch_layout_t>;
    using physics_t = amr::ndt::solver::physics_system<patch_index_t, patch_layout_t, {domain_size, domain_size}>;

    // --- 3. Instantiate Solver ---
    // Capacity=1000, Gamma=0 (unused), Courant=0.4
    amr_solver<tree_t, physics_t, DIM> solver(1000, 1.0, 0.4); 
    amr::ndt::print::vtk_print<physics_t> printer("advection_results");

    // --- 4. Define Initial Condition (Gaussian Pulse) ---
    auto gaussianPulseIC = [](double x, double y) -> amr::containers::static_vector<double, 1> {
        double dx = x - 0.2; // Start near bottom-left
        double dy = y - 0.2;
        double r2 = dx*dx + dy*dy;
        double val = std::exp(-r2 / 0.01); // Standard deviation sigma=0.1
        
        amr::containers::static_vector<double, 1> state;
        state[0] = val;
        return state;
    };

    // --- 5. AMR Criterion (Refine where scalar > 0.1) ---
    auto refinementLogic = [&](const patch_index_t& idx) {
        auto patch = solver.get_tree().template get_patch<0>(idx);
        double max_val = 0.0;
        for (auto v : patch) max_val = std::max(max_val, v);

        if (max_val > 0.1 && idx.level() < 5) return tree_t::refine_status_t::Refine;
        if (max_val < 0.05 && idx.level() > 1) return tree_t::refine_status_t::Coarsen;
        return tree_t::refine_status_t::Stable;
    };

    // --- 6. Initialization & Execution ---
    std::cout << "Initializing Advection Test...\n";
    solver.initialize(gaussianPulseIC);
    
    double t = 0.0;
    double t_max = 1.0;
    int step = 0;

    while (t < t_max) {
        double dt = solver.compute_time_step();
        solver.time_step(dt);
        
        t += dt;
        step++;

        if (step % 5 == 0) {
            solver.get_tree().reconstruct_tree(refinementLogic);
            solver.get_tree().halo_exchange_update();
        }

        if (step % 20 == 0) {
            printer.print(solver.get_tree(), "_step_" + std::to_string(step) + ".vtk");
            std::cout << "T: " << t << " | Patches: " << solver.get_tree().size() << "\n";
        }
    }

    return 0;
}