#include "../build/generated_config.hpp"
#include "containers/container_utils.hpp"
#include "containers/static_layout.hpp"
#include "containers/static_shape.hpp"
#include "containers/static_vector.hpp"
#include "dg_helpers/basis/basis.hpp"
#include "dg_helpers/equations/equations.hpp"
#include "dg_helpers/globals/global_config.hpp"
#include "dg_helpers/globals/globals.hpp"
#include "dg_helpers/rhs.hpp"
#include "dg_helpers/surface.hpp"
#include "dg_helpers/time_integration/time_integration.hpp"
#include "dg_helpers/tree_builder.hpp"
#include "morton/morton_id.hpp"
#include "ndtree/ndhierarchy.hpp"
#include "ndtree/ndtree.hpp"
#include "ndtree/patch_layout.hpp"
#include "ndtree/print_dg_tree_v2.hpp"
#include "utility/random.hpp"
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits> // for std::numeric_limits
#include <random>
#include <tuple>
#include <utility>

using namespace amr::equations;
using namespace amr::containers;
using namespace amr::config;
using namespace amr::global;
using namespace amr::time_integration;

/**
 * @brief DG Solver with Adaptive Mesh Refinement
 *
 * Demonstrates:
 * - Configuration-driven equation and mesh setup
 * - Tree-based AMR with Morton indexing
 * - Per-cell DOF storage and time-stepping
 * - RHS evaluation using finite volume/DG approach
 * - VTK output of DG field data
 */
int main()
{
    std::cout << "====================================\n";
    std::cout << "  DG Solver - AMR Main Loop\n";
    std::cout << "====================================\n\n";
    std::cout << "Configuration:\n";
    std::cout << "  Order=" << amr::config::GlobalConfigPolicy::Order
              << ", Dim=" << amr::config::GlobalConfigPolicy::Dim
              << ", DOFs=" << amr::config::GlobalConfigPolicy::DOFs << "\n";

    using global_t = GlobalConfig<amr::config::GlobalConfigPolicy>;
    using RHS      = amr::rhs::RHSEvaluator<global_t, amr::config::GlobalConfigPolicy>;
    using dg_tree  = amr::dg_tree::TreeBuilder<global_t, amr::config::GlobalConfigPolicy>;
    using S1       = dg_tree::S1;
    using S2       = dg_tree::S2;
    using S3       = dg_tree::S3;
    using patch_index_t = typename dg_tree::patch_index_t;

    dg_tree tree_builder;
    auto&   tree = tree_builder.tree;

    using patch_layout_t    = typename dg_tree::patch_layout_t;
    using patch_container_t = amr::containers::
        static_tensor<typename S1::type, typename patch_layout_t::padded_layout_t>;
    using IntegratorTraits = amr::time_integration::TimeIntegratorTraits<
        amr::config::GlobalConfigPolicy::integrator,
        patch_container_t>;

    typename IntegratorTraits::type integrator;

    double time         = 0.0;
    double dt           = 0.01;
    auto   max_eigenval = -std::numeric_limits<double>::infinity();
    int    timestep     = 0;
    double next_plotted = 0.0; // Start plotting at time 0

    // Track the current level being refined for selective bottom-left refinement
    int current_refine_level = 0; // Start refining from the root (level 0)

    // Define AMR refinement condition: refine only bottom-left patch (0,0) progressively
    auto amr_condition = [&current_refine_level](
                             const patch_index_t& idx,
                             [[maybe_unused]] int step,
                             [[maybe_unused]]
                             int max_step
                         )
    {
        auto [coords, level] = patch_index_t::decode(idx.id());
        auto max_depth       = idx.max_depth();

        // Only refine the (0,0) patch at the current level
        bool is_bottom_left = (coords[0] == 0 && coords[1] == 0);

        // Refine only if this patch is at the current level being refined
        if (is_bottom_left && level == current_refine_level && level < max_depth)
        {
            return dg_tree::tree_t::refine_status_t::Refine;
        }

        // All other patches remain stable, no coarsening
        return dg_tree::tree_t::refine_status_t::Stable;
    };

    try
    {
        ndt::print::dg_tree_printer_2d<global_t, GlobalConfigPolicy> printer("dg_tree");

        std::cout << "\n====================================\n";
        std::cout << "  Starting Time Integration\n";
        std::cout << "====================================\n\n";

        std::string time_extension = "_t" + std::to_string(timestep) + ".vtk";
        printer.template print<S1>(tree, time_extension);
        std::cout << "  Output: " << time_extension << " (timestep " << timestep << ")\n";

        double plot_step = 0.01; // Plot every 0.1 time units
        next_plotted     = 0.0;
        int amr_step     = 0;

        while (time < amr::config::GlobalConfigPolicy::EndTime)
        {
            // Apply mesh refinement/coarsening once per timestep (like dynamic_amr)
            auto amr_condition_with_time =
                [&amr_condition, &amr_step](const patch_index_t& idx)
            {
                return amr_condition(
                    idx,
                    amr_step,
                    static_cast<int>(amr::config::GlobalConfigPolicy::EndTime * 10)
                );
            };

            std::size_t tree_size_before = tree.size();
            tree.reconstruct_tree(amr_condition_with_time);
            std::size_t tree_size_after = tree.size();

            // If tree was refined, move to next level for selective refinement
            if (tree_size_after > tree_size_before &&
                current_refine_level < static_cast<int>(patch_index_t::max_depth()))
            {
                current_refine_level++;
            }

            // Initialize halo cells with periodic boundary conditions
            tree.halo_exchange_update();

            // Apply time integrator to each patch in the tree
            for (std::size_t idx = 0; idx < tree.size(); ++idx)
            {
                auto& dof_patch        = tree.template get_patch<S1>(idx);
                auto& flux_patch       = tree.template get_patch<S2>(idx);
                auto& center_patch     = tree.template get_patch<S3>(idx);
                auto  edge             = global_t::cell_edge(idx);
                auto  volume           = global_t::cell_volume(edge);
                auto  surface          = global_t::cell_area(edge);
                auto  inverse_jacobian = 1 / edge;

                // Skip dt calc on first step or when max_eigenval is invalid
                if (max_eigenval > 0)
                {
                    dt = 1 /
                         (amr::config::GlobalConfigPolicy::Order *
                              amr::config::GlobalConfigPolicy::Order +
                          1.0) *
                         amr::config::CourantNumber * edge / max_eigenval;
                }

                auto residual_callback = [&](
                                             patch_container_t&       patch_update,
                                             const patch_container_t& current_dofs,
                                             double /*t*/
                                         )
                {
                    RHS::evaluate(
                        current_dofs,
                        flux_patch.data(),
                        patch_update,
                        center_patch.data(),
                        dt,
                        volume,
                        surface,
                        max_eigenval,
                        inverse_jacobian
                    );
                };

                integrator.step(residual_callback, dof_patch.data(), time, dt);
            }

            time += dt;
            ++timestep;
            ++amr_step;

            if (time >= next_plotted - 1e-10)
            {
                time_extension = "_t" + std::to_string(timestep) + ".vtk";
                printer.template print<S1>(tree, time_extension);
                std::cout << "  Output: " << time_extension << " (timestep " << timestep
                          << ", time=" << time << ")\n";
                next_plotted += plot_step;
            }
        }
        std::cout << "dt: " << dt << "\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << "Exception caught: " << e.what() << "\n";
    }

    return 0;
}