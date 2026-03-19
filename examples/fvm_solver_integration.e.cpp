#include "containers/static_layout.hpp"
#include "containers/static_shape.hpp"
#include "containers/static_vector.hpp"
#ifdef AMR_ENABLE_CUDA_AMR
#    include "cuda/fvm_refinement_criterion.hpp"
#endif
#include "morton/morton_id.hpp"
#include "ndtree/intergrid_operator.hpp"
#include "ndtree/ndhierarchy.hpp"
#include "ndtree/ndtree.hpp"
#include "ndtree/patch_layout.hpp"
#include "ndtree/vtk_print.hpp"
#include "solver/EulerPhysics.hpp"
#include "solver/amr_solver.hpp"
#include "solver/cell_types.hpp"
#include "solver/physics_system.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>
#include <unordered_map>

int main()
{
    std::cout << "Hello AMR world\n";
    constexpr std::size_t N         = 10;
    constexpr std::size_t M         = 10;
    constexpr std::size_t Halo      = 2;
    constexpr double      physics_x = 1000;
    constexpr double      physics_y = 1000;

    constexpr std::array<double, 2> physics_lengths = { physics_x, physics_y };

    using shape_t  = amr::containers::static_shape<N, M>;
    using layout_t = amr::containers::static_layout<shape_t>;

    using patch_index_t  = amr::ndt::morton::morton_id<7u, 2u>;
    using patch_layout_t = amr::ndt::patches::patch_layout<layout_t, Halo>;
    using intergrid_operator_t =
        amr::ndt::intergrid_operator::linear_interpolator<patch_layout_t>;
    using tree_t = amr::ndt::tree::ndtree<
        amr::cell::EulerCell2D,
        patch_index_t,
        patch_layout_t,
        intergrid_operator_t>;
    using rho_patch_t = typename tree_t::template patch_t<amr::cell::Rho>;

    static_assert(
        sizeof(rho_patch_t) == patch_layout_t::flat_size() * sizeof(double),
        "Rho patch mirror must be a contiguous flat double buffer"
    );

    using physics_t =
        amr::ndt::solver::physics_system<patch_index_t, patch_layout_t, physics_lengths>;

    amr::ndt::print::vtk_print<physics_t> printer("euler_print");

    double tmax = 400; // Example tmax, adjust as needed
    [[maybe_unused]]
    double print_frequency = 5.0; // Print every 10 seconds
    [[maybe_unused]]
    const std::string output_prefix = "solver_integration_test_refine";

    int inital_refinement = 3;

    // Instantiate the AMR solver.
    amr_solver<tree_t, physics_t, EulerPhysics2D, 2> solver(
        1000000, 1.4, 0.3
    ); // Provide initial capacity for tree

    auto refineAll = [&]([[maybe_unused]]
                         const patch_index_t& idx)
    {
        return tree_t::refine_status_t::Refine;
    };

    [[maybe_unused]] auto acousticWaveCriterion = [&](const patch_index_t& idx)
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
            max_rho_value = std::max(rho_patch[linear_idx], max_rho_value);
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

#ifdef AMR_ENABLE_CUDA_AMR
    auto applyAcousticWaveCriterionGpu = [&]()
    {
        constexpr double REFINE_RHO_THRESHOLD  = 0.53;
        constexpr double COARSEN_RHO_THRESHOLD = 0.49;
        constexpr int    MAX_LEVEL             = 6;
        constexpr int    MIN_LEVEL             = 1;

        std::vector<int>    patch_levels(solver.get_tree().size());
        std::vector<std::int8_t> raw_decisions(solver.get_tree().size());
        std::vector<patch_index_t> patch_ids(solver.get_tree().size());

        for (std::size_t patch_linear_idx = 0; patch_linear_idx < solver.get_tree().size();
             ++patch_linear_idx)
        {
            const auto patch_id = solver.get_tree().get_node_index_at(patch_linear_idx);
            patch_ids[patch_linear_idx] = patch_id;
            patch_levels[patch_linear_idx] = static_cast<int>(patch_id.level());
        }

        const auto* rho_device_buffer = reinterpret_cast<const double*>(
            solver.get_tree().template get_device_buffer<amr::cell::Rho>()
        );

        amr::cuda::compute_scalar_patch_amr_decisions_from_device(
            rho_device_buffer,
            patch_levels.data(),
            patch_levels.size(),
            amr::cuda::scalar_patch_amr_launch_config{
                .num_patches        = solver.get_tree().size(),
                .cells_per_patch    = patch_layout_t::flat_size(),
                .refine_threshold   = REFINE_RHO_THRESHOLD,
                .coarsen_threshold  = COARSEN_RHO_THRESHOLD,
                .min_level          = MIN_LEVEL,
                .max_level          = MAX_LEVEL,
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

    // Parameters for the Acoustic Pulse
    constexpr double RHO_BG    = 0.5;
    constexpr double P_BG      = 1.0;
    constexpr double AMPLITUDE = 10.0;
    constexpr double PULSE_WIDTH_SQ =
        0.01 * physics_x * physics_y; // sigma^2 in physical units
    constexpr double CENTER_X = 0.5 * physics_x;
    constexpr double CENTER_Y = 0.5 * physics_y;

    // The initial condition function (auto IC = [](){})
    auto acousticPulseIC =
        [&](auto const& coords) -> amr::containers::static_vector<double, 4>
    {
        const double x = coords[0];
        const double y = coords[1];

        amr::containers::static_vector<double, 4> prim;

        // Calculate distance squared from the center
        double dx   = x - CENTER_X;
        double dy   = y - CENTER_Y;
        double r_sq = dx * dx + dy * dy;

        // Calculate the density/pressure perturbation
        double perturbation = AMPLITUDE * std::exp(-r_sq / PULSE_WIDTH_SQ);

        // Set primitive variables: [rho, u, v, p]
        prim[0] = RHO_BG + (perturbation * 0.2); // Density (rho)
        prim[1] = 0.0;                           // X-velocity (u)
        prim[2] = 0.0;                           // Y-velocity (v)
        prim[3] = P_BG + perturbation;           // Pressure (p)

        return prim;
    };

    std::cout << "Initializing solver..." << std::endl;
    for (int i = 0; i < inital_refinement; i++)
    {
        solver.get_tree().reconstruct_tree(refineAll);
        solver.get_tree().halo_exchange_update();
    }

    solver.initialize(acousticPulseIC);
    solver.get_tree().halo_exchange_update();
#ifdef AMR_ENABLE_CUDA_AMR
    solver.get_tree().sync_current_to_device();
#endif

    // Print initial state
    printer.print(solver.get_tree(), "_iteration_0.vtk");

    // Main Simulation Loop
    double t    = 0.0;
    int    step = 1;
    [[maybe_unused]]
    double next_print_time = print_frequency;
    [[maybe_unused]]
    int output_counter = 1;

    std::cout << "\nStarting AMR simulation...\n";

    std::size_t cell_update_count = 0;
    const auto  start             = std::chrono::steady_clock::now();

    while (t < tmax)
    {
        const double dt = solver.compute_time_step();

        DEFAULT_SOURCE_LOG_PROGRESS("Step: {},\tt: {:.5f},\tdt: {:.5f} ", step, t, dt);

        solver.time_step(dt);
        cell_update_count +=
            solver.get_tree().size() * patch_layout_t::data_layout_t::flat_size();
#ifdef AMR_ENABLE_CUDA_AMR
        solver.get_tree().sync_current_to_device();
#endif

        if (step % 5 == 0)
        {
#ifdef AMR_ENABLE_CUDA_AMR
            applyAcousticWaveCriterionGpu();
#else
            solver.get_tree().reconstruct_tree(acousticWaveCriterion);
#endif
            solver.get_tree().halo_exchange_update();
#ifdef AMR_ENABLE_CUDA_AMR
            solver.get_tree().sync_current_to_device();
#endif
        }

        t += dt;

        // Print only when we've passed the next print time
        if (t >= next_print_time)
        {
            std::string file_extension =
                "_iteration_" + std::to_string(output_counter) + ".vtk";
            printer.print(solver.get_tree(), file_extension);
            next_print_time += print_frequency;
            output_counter++;
            DEFAULT_SOURCE_LOG_INFO("Written vtk output: {}", file_extension);
        }

        step++;
    }
    const auto                          end      = std::chrono::steady_clock::now();
    const std::chrono::duration<double> duration = end - start;
    std::cout << "Updated cells: " << cell_update_count << '\n';
    std::cout << "Duration: " << duration.count() << '\n';
    std::cout << "Updates per second: " << (double)cell_update_count / duration.count()
              << '\n';

    std::cout << "\nSimulation completed. Files in vtk_output/ directory." << std::endl;

    return 0;
}
