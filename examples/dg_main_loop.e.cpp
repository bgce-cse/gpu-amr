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
#include <limits>
#include <random>
#include <tuple>
#include <utility>

using namespace amr::equations;
using namespace amr::containers;
using namespace amr::config;
using namespace amr::global;
using namespace amr::time_integration;

int main()
{
    std::cout << "====================================\n";
    std::cout << "  DG Solver - AMR Main Loop\n";
    std::cout << "====================================\n\n";

    using global_t = GlobalConfig<amr::config::GlobalConfigPolicy>;
    using RHS      = amr::rhs::RHSEvaluator<global_t, amr::config::GlobalConfigPolicy>;
    using dg_tree  = amr::dg_tree::TreeBuilder<global_t, amr::config::GlobalConfigPolicy>;
    using S1       = dg_tree::S1;
    using S2       = dg_tree::S2;

    using patch_index_t = typename dg_tree::patch_index_t;
    using tree_type_t   = typename dg_tree::tree_t;

    dg_tree tree_builder;
    auto&   tree = tree_builder.tree;

    using patch_layout_t    = typename dg_tree::patch_layout_t;
    using patch_container_t = amr::containers::
        static_tensor<typename S1::type, typename patch_layout_t::padded_layout_t>;
    using IntegratorTraits = amr::time_integration::TimeIntegratorTraits<
        amr::config::GlobalConfigPolicy::integrator,
        patch_container_t>;

    typename IntegratorTraits::type integrator;

    double time     = 0.0;
    double dt       = 0.01;
    double dt_patch = dt;
    int    timestep = 0;

    double plot_step    = 0.01;
    double next_plotted = 0.0;

    // CFL-scaled AMR schedule
    int  amr_interval         = 10;
    int  next_amr_step        = 0;
    int  amr_step             = 0;
    int  current_refine_level = 0;
    auto amr_condition = [&current_refine_level, &timestep](const patch_index_t& idx)
    {
        auto [coords, level] = patch_index_t::decode(idx.id());
        auto max_depth       = idx.max_depth();

        // Between timestep 20 and 50: coarsen all patches
        if (timestep >= 20 && timestep <= 50 && level > 0)
        {
            return tree_type_t::refine_status_t::Coarsen;
        }

        // Outside the coarsening window: refine normally, but only if not in coarsening
        // phase
        if (timestep < 20 || timestep > 50)
        {
            bool is_bottom_left = (coords[0] == 0 && coords[1] == 0);

            // Refine only if this patch is at the current level being refined
            if (is_bottom_left && level == current_refine_level && level < max_depth)
            {
                return tree_type_t::refine_status_t::Refine;
            }
        }

        // All other patches remain stable
        return tree_type_t::refine_status_t::Stable;
    };

    try
    {
        ndt::print::dg_tree_printer_2d<global_t, GlobalConfigPolicy> printer("dg_tree");

        std::string time_extension = "_t" + std::to_string(timestep) + ".vtk";
        printer.template print<S1>(tree, time_extension);

        auto max_eigenval = -std::numeric_limits<double>::infinity();
        while (time < amr::config::GlobalConfigPolicy::EndTime)
        {
            tree.halo_exchange_update();
            for (std::size_t idx = 0; idx < tree.size(); ++idx)
            {
                auto& dof_patch        = tree.template get_patch<S1>(idx);
                auto& flux_patch       = tree.template get_patch<S2>(idx);
                auto  edge             = global_t::cell_edge(tree.get_node_index_at(idx));
                auto  volume           = global_t::cell_volume(edge);
                auto  surface          = global_t::cell_area(edge);
                auto  inverse_jacobian = 1.0 / edge;

                // std::cout << "dt = " << dt << " for patch = " << idx
                //           << " eig = " << max_eigenval << " edge = " << edge
                //           << " volume = " << volume << "\n";

                auto residual_callback = [&](patch_container_t&       patch_update,
                                             const patch_container_t& current_dofs,
                                             double)
                {
                    RHS::evaluate(
                        current_dofs,
                        flux_patch.data(),
                        patch_update,
                        dt,
                        volume,
                        surface,
                        max_eigenval,
                        inverse_jacobian
                    );
                };

                integrator.step(residual_callback, dof_patch.data(), time, dt);

                if (max_eigenval > 0)
                {
                    double new_dt = 1.0 /
                                    (amr::config::GlobalConfigPolicy::Order *
                                         amr::config::GlobalConfigPolicy::Order +
                                     1.0) *
                                    amr::config::CourantNumber * edge / max_eigenval;
                    dt_patch = std::min(dt_patch, new_dt);
                }

                // if (timestep >= 40 && timestep <= 50)
                // {
                //     std::cout << "dioschifosocane" << dof_patch.data() << "\n\n";
                // }
            }
            dt = std::min(dt, dt_patch);

            // // Debug: Check for anomalies
            // if (timestep >= 40 && timestep <= 50)
            // {
            //     std::cout << "[DEBUG] Step " << timestep << ": dt=" << dt
            //               << ", max_eigenval=" << max_eigenval
            //               << ", tree.size()=" << tree.size() << std::endl;
            // }

            time += dt;
            ++timestep;

            // ================================
            // CFL-scaled AMR
            // ================================
            if (timestep >= next_amr_step)
            {
                ++amr_step;

                tree.reconstruct_tree(amr_condition);

                // Progress to next refinement level
                if (current_refine_level < static_cast<int>(patch_index_t::max_depth()))
                {
                    current_refine_level++;
                }

                next_amr_step = timestep + amr_interval;
            }

            // ================================
            // Output (independent of AMR)
            // ================================
            if (true)
            {
                time_extension = "_t" + std::to_string(timestep) + ".vtk";
                printer.template print<S1>(tree, time_extension);
                std::cout << "Output " << time_extension << " time=" << time << "\n";
                next_plotted += plot_step;
            }
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Exception: " << e.what() << "\n";
    }

    return 0;
}
