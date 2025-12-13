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
#include <iomanip> // for std::setw, std::setprecision
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
    std::cout << "  Order=" << amr::config::Order << ", Dim=" << amr::config::Dim
              << ", DOFs=" << amr::config::DOFs << "\n";
    std::cout << "  Equation=" << amr::config::Equation << "\n\n";

    using global_t = GlobalConfig<amr::config::GlobalConfigPolicy>;
    using RHS      = amr::rhs::RHSEvaluator<global_t, amr::config::GlobalConfigPolicy>;
    using dg_tree  = amr::dg_tree::TreeBuilder<global_t, amr::config::GlobalConfigPolicy>;
    using S1       = dg_tree::S1;
    using S2       = dg_tree::S2;
    using S3       = dg_tree::S3;
    // using S4      = dg_tree::S4;
    dg_tree tree_builder;
    auto&   tree = tree_builder.tree;

    using patch_layout_t    = typename dg_tree::patch_layout_t;
    using patch_container_t = amr::containers::
        static_tensor<typename S1::type, typename patch_layout_t::padded_layout_t>;
    using IntegratorTraits = amr::time_integration::
        TimeIntegratorTraits<amr::config::time_integrator, patch_container_t>;
    typename IntegratorTraits::type integrator;

    // Setup tree mesh
    // Time-stepping loop
    double time     = 0.0;
    double dt       = 0.01; // TODO: CFL condition based on max eigenvalue
    int    timestep = 0;

    try
    {
        std::cout << "\n====================================\n";
        std::cout << "  Initializing DG Tree Printer (Refactored)\n";
        std::cout << "====================================\n\n";
        ndt::print::dg_tree_printer_refactored<global_t> printer("dg_tree");
        std::cout << "DG tree printer created successfully\n";

        std::cout << "\n====================================\n";
        std::cout << "  Starting Time Integration\n";
        std::cout << "====================================\n\n";
        std::string time_extension = "_t" + std::to_string(timestep) + ".vtk";
        printer.template print<S1>(tree, time_extension);
        std::cout << "  Output: " << time_extension << " (timestep " << timestep << ")\n";

        while (time < amr::config::EndTime)
        {
            // Initialize halo cells with periodic boundary conditions
            tree.halo_exchange_update();
            // Apply time integrator to each patch in the tree
            for (std::size_t idx = 0; idx < tree.size(); ++idx)
            {
                auto& dof_patch    = tree.template get_patch<S1>(idx);
                auto& flux_patch   = tree.template get_patch<S2>(idx);
                auto& center_patch = tree.template get_patch<S3>(idx);

                auto residual_callback = [&](patch_container_t&       patch_update,
                                             const patch_container_t& current_dofs,
                                             double /*t*/)
                {
                    // std::cout << "global values" << globals.basis.quadpoints()
                    //           << " quadweights " << globals.basis.quadweights() <<
                    //           "\n";
                    RHS::evaluate(
                        current_dofs,
                        flux_patch.data(),
                        patch_update,
                        center_patch.data(),
                        dt,
                        patch_layout_t{},
                        0.01
                    );
                };

                integrator.step(residual_callback, dof_patch.data(), time, dt);
            }

            // Update halo cells with periodic boundary conditions
            tree.halo_exchange_update();
            if (timestep % 5 == 4)
            {
                time_extension = "_t" + std::to_string(timestep) + ".vtk";
                printer.template print<S1>(tree, time_extension);
                std::cout << "  Output: " << time_extension << " (timestep " << timestep
                          << ")\n";
            }

            // Advance time
            time += dt;
            ++timestep;
        }

        // TODO: Generate PVD file for time series visualization
        // ndt::print::dg_tree_printer<Dim, Order, PatchSize, HaloWidth, DOFs> printer(
        //     "dg_tree_timestep"
        // );
        // printer.generate_pvd_file(
        //     "vtk_output/dg_tree_advanced_simulation.pvd", vtk_files, times
        // );
        // std::cout << "PVD file generated successfully\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << "Exception caught: " << e.what() << "\n";
    }

    return 0;
}