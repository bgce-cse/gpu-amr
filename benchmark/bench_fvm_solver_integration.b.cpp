#include "containers/static_layout.hpp"
#include "containers/static_shape.hpp"
#include "containers/static_vector.hpp"
#ifdef AMR_ENABLE_CUDA_AMR
#    include "cuda/fvm_refinement_criterion.hpp"
#    include "cuda/profiler.hpp"
#endif
#include "morton/morton_id.hpp"
#include "ndtree/intergrid_operator.hpp"
#include "ndtree/ndhierarchy.hpp"
#include "ndtree/ndtree.hpp"
#include "ndtree/patch_layout.hpp"
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

#define VTK_PRINT (false)

int main()
{
    std::cout << "Hello AMR world\n";
#ifdef AMR_ENABLE_CUDA_AMR
    std::cout << "CUDA ENABLED\n";
#else
    std::cout << "CUDA DISABLED\n";
#endif
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

    using physics_t =
        amr::ndt::solver::physics_system<patch_index_t, patch_layout_t, physics_lengths>;

#if VTK_PRINT
    amr::ndt::print::vtk_print<physics_t> printer("euler_print");
#endif

    double tmax = 400; // Example tmax, adjust as needed
    [[maybe_unused]]
    double print_frequency = 5.0; // Print every 10 seconds
    [[maybe_unused]]
    const std::string output_prefix = "solver_integration_test_refine";

    int inital_refinement = 3;

    // Instantiate the AMR solver.
    amr_solver<tree_t, physics_t, EulerPhysics2D, 2> solver(
        10000, 1.4, 0.3
    ); // Provide initial capacity for tree

    auto refineAll = [&]([[maybe_unused]]
                         const patch_index_t& idx)
    {
        return tree_t::refine_status_t::Refine;
    };

    struct acoustic_wave_amr_criterion
    {
        tree_t& tree;

        double s_refine_threshold  = 0.53;
        double s_coarsen_threshold = 0.49;
        int    s_max_level         = 6;
        int    s_min_level         = 1;

        auto operator()(const patch_index_t& idx) const -> tree_t::refine_status_t
        {
            const int level = idx.level();
            auto      rho_patch = tree.template get_patch<amr::cell::Rho>(idx);
            double    max_rho_value = 0;

            for (std::size_t linear_idx = 0; linear_idx != patch_layout_t::flat_size();
                 ++linear_idx)
            {
                max_rho_value = std::max(rho_patch[linear_idx], max_rho_value);
            }

            if (level < s_max_level && max_rho_value > s_refine_threshold)
            {
                return tree_t::refine_status_t::Refine;
            }

            if (level > s_min_level && max_rho_value < s_coarsen_threshold)
            {
                return tree_t::refine_status_t::Coarsen;
            }

            return tree_t::refine_status_t::Stable;
        }

#ifdef AMR_ENABLE_CUDA_AMR
        auto fill_refine_flags(tree_t& target_tree) const -> void
        {
            const auto* rho_device_buffer = reinterpret_cast<const double*>(
                target_tree.template get_device_buffer<amr::cell::Rho>()
            );

            amr::cuda::compute_scalar_patch_amr_decisions_from_device(
                rho_device_buffer,
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

    const auto acousticWaveCriterion = acoustic_wave_amr_criterion{ solver.get_tree() };

    // Parameters for the Acoustic Pulse
    constexpr double RHO_BG    = 0.5;
    constexpr double P_BG      = 1.0;
    constexpr double AMPLITUDE = 10.0;
    constexpr double PULSE_WIDTH_SQ =
        0.01 * physics_x * physics_y; // sigma^2 in physical units
    constexpr double CENTER_X = 0.5 * physics_x;
    constexpr double CENTER_Y = 0.5 * physics_y;

    // The initial condition function
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
#ifdef AMR_ENABLE_CUDA_AMR
        solver.get_tree().sync_current_to_device();
#endif
        solver.get_tree().halo_exchange_update();
#ifdef AMR_ENABLE_CUDA_AMR
        solver.get_tree().sync_current_from_device();
#endif
    }

    solver.initialize(acousticPulseIC);
#ifdef AMR_ENABLE_CUDA_AMR
    solver.get_tree().sync_current_to_device();
    solver.get_tree().build_patch_levels_on_device();
#endif
    solver.get_tree().halo_exchange_update();

    // Main Simulation Loop
    double t    = 0.0;
    int    step = 1;
    [[maybe_unused]]
    double next_print_time = print_frequency;
    [[maybe_unused]]
    int output_counter = 1;

    std::cout << "\nStarting AMR simulation...\n";

    std::size_t cell_update_count = 0;
    constexpr int reconstruction_interval = 10;
#ifdef AMR_ENABLE_CUDA_AMR
    amr::cuda::profile_capture_start();
#endif
    const auto  start             = std::chrono::steady_clock::now();

    while (t < tmax)
    {
        solver.advance_batch_async(reconstruction_interval);
        cell_update_count +=
            static_cast<std::size_t>(reconstruction_interval) *
            solver.get_tree().size() * patch_layout_t::data_layout_t::flat_size();

        solver.get_tree().reconstruct_tree(acousticWaveCriterion);
        solver.get_tree().halo_exchange_update();

        t += solver.finish_advance_batch();
        step += reconstruction_interval;
    }
#ifdef AMR_ENABLE_CUDA_AMR
    amr::cuda::profile_capture_stop();
#endif
    const auto                          end      = std::chrono::steady_clock::now();
    const std::chrono::duration<double> duration = end - start;
    std::cout << "Updated cells: " << cell_update_count << '\n';
    std::cout << "Duration: " << duration.count() << '\n';
    std::cout << "Updates per second: " << (double)cell_update_count / duration.count()
              << '\n';

    std::cout << "\nSimulation completed. Files in vtk_output/ directory." << std::endl;

    return 0;
}
