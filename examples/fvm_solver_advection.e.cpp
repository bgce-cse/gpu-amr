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
#include <unordered_map>

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
    [[maybe_unused]] auto amrCriterion = [&](const patch_index_t& idx)
    {
        auto   patch   = solver.get_tree().template get_patch<amr::cell::Scalar>(idx);
        auto&  data    = patch.data();
        double max_val = 0;
        for (auto v : data)
        {
            max_val = std::max(max_val, v);
        }

        if (max_val > 0.1 && idx.level() < 5) return tree_t::refine_status_t::Refine;
        if (max_val < 0.05 && idx.level() > 1) return tree_t::refine_status_t::Coarsen;
        return tree_t::refine_status_t::Stable;
    };

#ifdef AMR_ENABLE_CUDA_AMR
    auto applyAmrCriterionGpu = [&]()
    {
        constexpr double REFINE_THRESHOLD  = 0.1;
        constexpr double COARSEN_THRESHOLD = 0.05;
        constexpr int    MAX_LEVEL         = 5;
        constexpr int    MIN_LEVEL         = 1;

        std::vector<int>            patch_levels(solver.get_tree().size());
        std::vector<std::int8_t>    raw_decisions(solver.get_tree().size());
        std::vector<patch_index_t>  patch_ids(solver.get_tree().size());

        for (std::size_t patch_linear_idx = 0; patch_linear_idx < solver.get_tree().size();
             ++patch_linear_idx)
        {
            const auto patch_id = solver.get_tree().get_node_index_at(patch_linear_idx);
            patch_ids[patch_linear_idx] = patch_id;
            patch_levels[patch_linear_idx] = static_cast<int>(patch_id.level());
        }

        const auto* scalar_device_buffer = reinterpret_cast<const double*>(
            solver.get_tree().template get_device_buffer<amr::cell::Scalar>()
        );

        amr::cuda::compute_scalar_patch_amr_decisions_from_device(
            scalar_device_buffer,
            patch_levels.data(),
            patch_levels.size(),
            amr::cuda::scalar_patch_amr_launch_config{
                .num_patches       = solver.get_tree().size(),
                .cells_per_patch   = patch_layout_t::flat_size(),
                .refine_threshold  = REFINE_THRESHOLD,
                .coarsen_threshold = COARSEN_THRESHOLD,
                .min_level         = MIN_LEVEL,
                .max_level         = MAX_LEVEL,
            },
            raw_decisions.data(),
            raw_decisions.size()
        );

        std::unordered_map<patch_index_t, tree_t::refine_status_t> decision_map;
        decision_map.reserve(raw_decisions.size());
        for (std::size_t i = 0; i < raw_decisions.size(); ++i)
        {
            decision_map.emplace(
                patch_ids[i], static_cast<tree_t::refine_status_t>(raw_decisions[i])
            );
        }

        solver.get_tree().reconstruct_tree(
            [&decision_map](const patch_index_t& pid) -> tree_t::refine_status_t
            {
                if (const auto it = decision_map.find(pid); it != decision_map.end())
                {
                    return it->second;
                }
                return tree_t::refine_status_t::Stable;
            }
        );
    };
#endif

    // --- 6. Initialization & Execution ---
    std::cout << "Initializing Advection Test...\n";
    solver.initialize(gaussianIC);
    solver.get_tree().halo_exchange_update();
#ifdef AMR_ENABLE_CUDA_AMR
    solver.get_tree().sync_current_to_device();
#endif

    double t     = 0.0;
    double t_max = 3.0;
    int    step  = 0;

    while (t < t_max)
    {
        double dt = solver.compute_time_step();

        std::cout << "Step " << step << ", t=" << t << ", dt=" << dt << std::endl;

        solver.time_step(dt);
// #ifdef AMR_ENABLE_CUDA_AMR
//         solver.get_tree().sync_current_to_device();
// #endif
        solver.get_tree().halo_exchange_update();

        t += dt;
        step++;

        if (step % 5 == 0)
        {
#ifdef AMR_ENABLE_CUDA_AMR
            applyAmrCriterionGpu();
#else
            solver.get_tree().reconstruct_tree(amrCriterion);
#endif
            solver.get_tree().halo_exchange_update();
#ifdef AMR_ENABLE_CUDA_AMR
            solver.get_tree().sync_current_to_device();
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
