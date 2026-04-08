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
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

int main()
{
    std::cout << "Hello AMR world\n";
    std::cout << "Benchmark variant: active AMR topology\n";
#ifdef AMR_ENABLE_CUDA_AMR
    std::cout << "CUDA ENABLED\n";
#else
    std::cout << "CUDA DISABLED\n";
#endif
    constexpr std::size_t N         = 64;
    constexpr std::size_t M         = 64;
    constexpr std::size_t Halo      = 1;
    constexpr double      physics_x = 1000;
    constexpr double      physics_y = 1000;
    double                tmax      = 30;
    int                   initial_refinement = 2;

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

    amr_solver<tree_t, physics_t, EulerPhysics2D, 2> solver(
        10000, 1.4, 0.3
    );

    auto refine_all = [&]([[maybe_unused]] const patch_index_t& idx)
    {
        return tree_t::refine_status_t::Refine;
    };

    struct active_wave_amr_criterion
    {
        tree_t& tree;

        double s_refine_threshold  = 0.512;
        double s_coarsen_threshold = 0.506;
        int    s_max_level         = 7;
        int    s_min_level         = 1;

        auto operator()(const patch_index_t& idx) const -> tree_t::refine_status_t
        {
            const int level = idx.level();
            auto      rho_patch = tree.template get_patch<amr::cell::Rho>(idx);
            double    max_rho_value = 0.0;

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

    const auto activeWaveCriterion = active_wave_amr_criterion{ solver.get_tree() };
    auto capture_topology = [&](const tree_t& tree) -> std::vector<patch_index_t>
    {
        std::vector<patch_index_t> topology;
        topology.reserve(tree.size());
        for (std::size_t i = 0; i != tree.size(); ++i)
        {
            topology.push_back(tree.get_node_index_at(i));
        }
        return topology;
    };

    constexpr double RHO_BG          = 0.5;
    constexpr double P_BG            = 1.0;
    constexpr double AMPLITUDE_A     = 12.0;
    constexpr double AMPLITUDE_B     = 9.0;
    constexpr double PULSE_WIDTH_SQ  = 0.004 * physics_x * physics_y;
    constexpr double CENTER_AX       = 0.35 * physics_x;
    constexpr double CENTER_AY       = 0.45 * physics_y;
    constexpr double CENTER_BX       = 0.68 * physics_x;
    constexpr double CENTER_BY       = 0.58 * physics_y;

    auto multi_pulse_ic =
        [&](auto const& coords) -> amr::containers::static_vector<double, 4>
    {
        const double x = coords[0];
        const double y = coords[1];

        amr::containers::static_vector<double, 4> prim;

        const double dx_a = x - CENTER_AX;
        const double dy_a = y - CENTER_AY;
        const double dx_b = x - CENTER_BX;
        const double dy_b = y - CENTER_BY;

        const double pulse_a =
            AMPLITUDE_A * std::exp(-(dx_a * dx_a + dy_a * dy_a) / PULSE_WIDTH_SQ);
        const double pulse_b =
            AMPLITUDE_B * std::exp(-(dx_b * dx_b + dy_b * dy_b) / PULSE_WIDTH_SQ);
        const double perturbation = pulse_a + pulse_b;

        prim[0] = RHO_BG + (perturbation * 0.2);
        prim[1] = 0.0;
        prim[2] = 0.0;
        prim[3] = P_BG + perturbation;

        return prim;
    };

    std::cout << "Initializing solver..." << std::endl;
    for (int i = 0; i < initial_refinement; ++i)
    {
        solver.get_tree().reconstruct_tree(refine_all);
#ifdef AMR_ENABLE_CUDA_AMR
        solver.get_tree().sync_current_to_device();
#endif
        solver.get_tree().halo_exchange_update();
#ifdef AMR_ENABLE_CUDA_AMR
        solver.get_tree().sync_current_from_device();
#endif
    }

    solver.initialize(multi_pulse_ic);
#ifdef AMR_ENABLE_CUDA_AMR
    solver.get_tree().sync_current_to_device();
    solver.get_tree().build_patch_levels_on_device();
#endif
    solver.get_tree().halo_exchange_update();

    double t    = 0.0;
    int    step = 1;

    std::cout << "\nStarting AMR simulation...\n";

    std::size_t cell_update_count                     = 0;
    std::size_t total_solver_timesteps                = 0;
    std::size_t initial_reconstruction_count          = static_cast<std::size_t>(initial_refinement);
    std::size_t timed_reconstruction_count            = 0;
    std::size_t identity_reconstruction_count         = 0;
    std::size_t topology_changed_reconstruction_count = 0;
    std::size_t min_patch_count                       = solver.get_tree().size();
    std::size_t max_patch_count                       = solver.get_tree().size();
    constexpr int reconstruction_interval             = 2;
#ifdef AMR_ENABLE_CUDA_AMR
    amr::cuda::profile_capture_start();
#endif
    const auto start = std::chrono::steady_clock::now();

    while (t < tmax)
    {
        const auto remaining_time = tmax - t;
        const auto patch_count_before_reconstruction = solver.get_tree().size();
        solver.advance_batch_async(reconstruction_interval, remaining_time);

        const auto topology_before_reconstruction = capture_topology(solver.get_tree());
        solver.get_tree().reconstruct_tree(activeWaveCriterion);
        const auto topology_unchanged =
            topology_before_reconstruction == capture_topology(solver.get_tree());
        solver.get_tree().halo_exchange_update();

        ++timed_reconstruction_count;
        if (topology_unchanged)
        {
            ++identity_reconstruction_count;
        }
        else
        {
            ++topology_changed_reconstruction_count;
        }

        std::size_t executed_steps = 0;
        t += solver.finish_advance_batch(&executed_steps);
        cell_update_count += executed_steps * patch_count_before_reconstruction *
                             patch_layout_t::data_layout_t::flat_size();
        total_solver_timesteps += executed_steps;
        step += static_cast<int>(executed_steps);
        min_patch_count = std::min(min_patch_count, solver.get_tree().size());
        max_patch_count = std::max(max_patch_count, solver.get_tree().size());
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
    std::cout << "Solver timesteps: " << total_solver_timesteps << '\n';
    std::cout << "Initial reconstructions: " << initial_reconstruction_count << '\n';
    std::cout << "Timed reconstructions: " << timed_reconstruction_count << '\n';
    std::cout << "Identity reconstructions: " << identity_reconstruction_count << '\n';
    std::cout << "Topology-changing reconstructions: "
              << topology_changed_reconstruction_count << '\n';
    std::cout << "Final patch count: " << solver.get_tree().size() << '\n';
    std::cout << "Min patch count: " << min_patch_count << '\n';
    std::cout << "Max patch count: " << max_patch_count << '\n';

    std::cout << "\nSimulation completed. Files in vtk_output/ directory." << std::endl;

    return 0;
}
