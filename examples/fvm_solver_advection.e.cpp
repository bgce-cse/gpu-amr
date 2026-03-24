#include "containers/static_layout.hpp"
#include "containers/static_shape.hpp"
#include "containers/static_vector.hpp"
#ifdef AMR_ENABLE_CUDA_AMR
#    include "cuda/fvm_refinement_criterion.hpp"
#endif
#include "morton/morton_id.hpp"
#include "ndtree/ndhierarchy.hpp"
#include "ndtree/ndtree.hpp"
#include "ndtree/patch_layout.hpp"
#include "ndtree/vtk_print.hpp"
#include "solver/AdvectionPhysics.hpp"
#include "solver/amr_solver.hpp"
#include "solver/cell_types.hpp"
#include "solver/physics_system.hpp"
#include <cmath>
#include <iostream>

int main()
{
    std::cout << "Hello Advection world\n";

    // --- 1. Simulation Parameters ---
    constexpr int         DIM         = 2;
    constexpr std::size_t N           = 10;
    constexpr std::size_t Halo        = 2;
    constexpr double      domain_size = 1.0;

    constexpr std::array<double, 2> physics_lengths = { domain_size, domain_size };

    using Cell = amr::cell::AdvectionCell;

    // --- 2. Boilerplate Types ---
    using shape_t        = amr::containers::static_shape<N, N>;
    using layout_t       = amr::containers::static_layout<shape_t>;
    using patch_index_t  = amr::ndt::morton::morton_id<7u, 2u>;
    using patch_layout_t = amr::ndt::patches::patch_layout<layout_t, Halo>;
    using intergrid_operator_t =
        amr::ndt::intergrid_operator::linear_interpolator<patch_layout_t>;

    using tree_t = amr::ndt::tree::ndtree<Cell, patch_index_t, patch_layout_t, intergrid_operator_t>;
    using scalar_patch_t = typename tree_t::template patch_t<amr::cell::Scalar>;

    static_assert(
        sizeof(scalar_patch_t) == patch_layout_t::flat_size() * sizeof(double),
        "Scalar patch mirror must be a contiguous flat double buffer"
    );
    using physics_t =
        amr::ndt::solver::physics_system<patch_index_t, patch_layout_t, physics_lengths>;

    // --- 3. Instantiate Solver ---
    // Capacity=1000, Gamma=0 (unused), Courant=0.4
    amr_solver<tree_t, physics_t, AdvectionPhysics<2>, DIM> solver(1000, 1.0, 0.4);
    amr::ndt::print::vtk_print<physics_t>                   printer("advection_test");

    // --- 4. Define Initial Condition (Gaussian Pulse) ---
    auto gaussianIC = [](auto const& coords) -> amr::containers::static_vector<double, 1>
    {
        double dx = coords[0] - 0.2;
        double dy = coords[1] - 0.2;
        double r2 = dx * dx + dy * dy;
        return { std::exp(-r2 / 0.005) };
    };

    // --- 5. AMR Criterion (Refine where scalar > 0.1) ---
    struct advection_amr_criterion
    {
        tree_t& tree;

        double s_refine_threshold  = 0.1;
        double s_coarsen_threshold = 0.05;
        int    s_max_level         = 5;
        int    s_min_level         = 1;

        auto operator()(const patch_index_t& idx) const -> tree_t::refine_status_t
        {
            auto   patch   = tree.template get_patch<amr::cell::Scalar>(idx);
            auto&  data    = patch.data();
            double max_val = 0;
            for (auto v : data)
            {
                max_val = std::max(max_val, v);
            }

            if (max_val > s_refine_threshold && idx.level() < s_max_level)
                return tree_t::refine_status_t::Refine;
            if (max_val < s_coarsen_threshold && idx.level() > s_min_level)
                return tree_t::refine_status_t::Coarsen;
            return tree_t::refine_status_t::Stable;
        }

#ifdef AMR_ENABLE_CUDA_AMR
        auto fill_refine_flags(tree_t& target_tree) const -> void
        {
            target_tree.sync_current_to_device();
            target_tree.build_patch_levels_on_device();

            const auto* scalar_device_buffer = reinterpret_cast<const double*>(
                target_tree.template get_device_buffer<amr::cell::Scalar>()
            );

            amr::cuda::compute_scalar_patch_amr_decisions_from_device(
                scalar_device_buffer,
                target_tree.get_device_patch_level_buffer(),
                target_tree.size(),
                amr::cuda::scalar_patch_amr_launch_config{
                    .num_patches       = target_tree.size(),
                    .cells_per_patch   = patch_layout_t::flat_size(),
                    .refine_threshold  = s_refine_threshold,
                    .coarsen_threshold = s_coarsen_threshold,
                    .min_level         = s_min_level,
                    .max_level         = s_max_level,
                },
                reinterpret_cast<std::int8_t*>(target_tree.get_device_refine_status_buffer()),
                target_tree.size()
            );

            target_tree.sync_refine_status_from_device();
        }
#endif
    };

    const auto amrCriterion = advection_amr_criterion{ solver.get_tree() };

    // --- 6. Initialization & Execution ---
    std::cout << "Initializing Advection Test...\n";
    solver.initialize(gaussianIC);
#ifdef AMR_ENABLE_CUDA_AMR
    solver.get_tree().sync_current_to_device();
    solver.get_tree().build_patch_levels_on_device();
#endif
    solver.get_tree().halo_exchange_update();
#ifdef AMR_ENABLE_CUDA_AMR
    solver.get_tree().sync_current_from_device();
#endif

    double t     = 0.0;
    double t_max = 3.0;
    int    step  = 0;

    while (t < t_max)
    {
        double dt = solver.compute_time_step();

        std::cout << "Step " << step << ", t=" << t << ", dt=" << dt << std::endl;

        solver.time_step(dt);

#ifdef AMR_ENABLE_CUDA_AMR
        solver.get_tree().sync_current_from_device();
#endif

        t += dt;
        step++;

        if (step % 5 == 0)
        {
            solver.get_tree().reconstruct_tree(amrCriterion);
            #ifdef AMR_ENABLE_CUDA_AMR
            solver.get_tree().sync_current_to_device();       // Send new tree to GPU
            solver.get_tree().build_patch_levels_on_device(); // Send new AMR levels
#endif
            solver.get_tree().halo_exchange_update();         // GPU exchanges boundaries
#ifdef AMR_ENABLE_CUDA_AMR
            solver.get_tree().sync_current_from_device();     // Bring back to CPU
#endif
        }

        if (step % 20 == 0)
        {
            printer.print(solver.get_tree(), "_step_" + std::to_string(step) + ".vtk");
            std::cout << "T: " << t << " | Patches: " << solver.get_tree().size() << "\n";
        }
    }

    std::cout << "\nSimulation completed. Files in vtk_output/ directory." << std::endl;

    return 0;
}
