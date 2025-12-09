#include "../build/generated_config.hpp"
#include "containers/container_utils.hpp"
#include "containers/static_layout.hpp"
#include "containers/static_shape.hpp"
#include "containers/static_vector.hpp"
#include "dg_helpers/basis.hpp"
#include "dg_helpers/equations.hpp"
#include "dg_helpers/equations/advection.hpp"
#include "dg_helpers/globals.hpp"
#include "dg_helpers/rhs.hpp"
#include "dg_helpers/surface.hpp"
#include "dg_helpers/time.hpp"
#include "morton/morton_id.hpp"
#include "ndtree/ndhierarchy.hpp"
#include "ndtree/ndtree.hpp"
#include "ndtree/patch_layout.hpp"
#include "ndtree/print_dg_tree_advanced.hpp"
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
using namespace amr::Basis;
using namespace amr::containers;
using namespace amr::config;
using namespace amr::global;
using namespace amr::time_integration;

struct S1
{
    using dof_value_t = static_vector<double, DOFs>;
    using type = typename utils::types::tensor::hypercube_t<dof_value_t, Order, Dim>;

    static constexpr auto index() noexcept -> std::size_t
    {
        return 0;
    }

    type value;
};

struct S2
{
    using dof_value_t = static_vector<double, DOFs>;
    using dof_t = typename utils::types::tensor::hypercube_t<dof_value_t, Order, Dim>;
    using type  = static_vector<dof_t, Dim>;

    static constexpr auto index() noexcept -> std::size_t
    {
        return 1;
    }

    type value;
};

struct S3
{
    using type = static_vector<double, Dim>;

    static constexpr auto index() noexcept -> std::size_t
    {
        return 2;
    }

    type value;
};

struct S4
{
    using type = static_vector<double, Dim>;

    static constexpr auto index() noexcept -> std::size_t
    {
        return 3;
    }

    type value;
}; // End of S4

// Marker cell type for tree AMR with DG data (DOF tensor, flux, center)

struct MarkerCell
{
    using deconstructed_types_map_t = std::tuple<S1, S2, S3, S4>;

    MarkerCell() = default;

    auto data_tuple() -> auto&
    {
        return m_data;
    }

    auto data_tuple() const -> auto const&
    {
        return m_data;
    }

    deconstructed_types_map_t m_data;
};

auto operator<<(std::ostream& os, MarkerCell const&) -> std::ostream&
{
    return os << "S1(DOF), S2(Flux), S3(Center), S4(Size)";
}

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

    // Configuration
    std::cout << "Configuration:\n";
    std::cout << "  Order=" << Order << ", Dim=" << Dim << ", DOFs=" << DOFs << "\n";
    std::cout << "  Equation=" << Equation << "\n\n";

    // Initialize global configuration
    constexpr std::size_t PatchSize = GridElements;
    constexpr std::size_t HaloWidth = 1;
    constexpr std::size_t MaxDepth  = 1;

    auto global_config = amr::global::GlobalConfig<Order, Dim, PatchSize, HaloWidth>();
    std::cout << "Global Config initialized:\n";
    std::cout << "  Basis quadweights: " << global_config.basis.quadweights() << "\n";
    std::cout << "  Mass tensor shape: " << global_config.mass_tensors.mass_tensor
              << "\n";
    std::cout << "  Inv mass tensor shape: " << global_config.mass_tensors.inv_mass_tensor
              << "\n";
    std::cout << "  Surface mass tensor shape: "
              << global_config.surface_mass_tensors.mass_tensor << "\n";

    // Initialize basis and equation (now available from global config)
    auto eq = make_configured_equation();

    // Create interpolator for initial DOF values
    auto interpolator = amr::equations::InitialDOFInterpolator(*eq, global_config.basis);

    // Setup tree mesh
    // Time-stepping loop
    double time     = 0.0;
    double dt       = 0.01; // TODO: CFL condition based on max eigenvalue
    int    timestep = 0;

    using shape_t        = static_shape<PatchSize, PatchSize>;
    using layout_t       = static_layout<shape_t>;
    using patch_index_t  = amr::ndt::morton::morton_id<MaxDepth, Dim>;
    using patch_layout_t = amr::ndt::patches::patch_layout<layout_t, HaloWidth>;
    using tree_t    = amr::ndt::tree::ndtree<MarkerCell, patch_index_t, patch_layout_t>;
    using Evaluator = amr::rhs::RHSEvaluator<Order, Dim, PatchSize, HaloWidth, DOFs>;

    tree_t tree(10000);

    using patch_container_t = decltype(tree.template get_patch<S1>(0).data());
    auto integrator         = make_configured_time_integrator<patch_container_t>();

    // Tree initialized with root patch having periodic boundaries (set in ndtree
    // constructor)
    std::cout << "Tree: MaxDepth=" << MaxDepth << ", Patches=" << tree.size()
              << " (single root patch with periodic BC)\n\n";

    // Initialize DG patches for each leaf
    // Populate leaf data
    for (std::size_t idx = 0; idx < tree.size(); ++idx)
    {
        auto& dof_patch          = tree.template get_patch<S1>(idx);
        auto& flux_patch         = tree.template get_patch<S2>(idx);
        auto& center_coord_patch = tree.template get_patch<S3>(idx);
        auto& size_patch         = tree.template get_patch<S4>(idx);

        auto patch_id                    = patch_index_t(idx);
        auto [patch_coords, patch_level] = patch_index_t::decode(patch_id.id());
        double patch_level_size          = 1.0 / static_cast<double>(1u << patch_level);
        double cell_size = patch_level_size / static_cast<double>(PatchSize);

        for (std::size_t linear_idx = 0; linear_idx != patch_layout_t::flat_size();
             ++linear_idx)
        {
            // Compute global cell center using generic function
            // std::cout << "Initializing patch " << idx << " cell linear_idx " <<
            // linear_idx
            //           << "is it halo? "
            //           << amr::ndt::utils::patches::is_halo_cell<patch_layout_t>(
            //                  linear_idx
            //              )
            //           << "patch coords: " << patch_coords[0] << "," << patch_coords[1]
            //           << " patch level: " << static_cast<int>(patch_level) << "\n";
            if (amr::ndt::utils::patches::is_halo_cell<patch_layout_t>(linear_idx))
                continue;
            // debug print

            // Convert linear index to local coordinates (dimension-generic)
            auto coords_with_halo =
                linear_to_local_coords<Dim, PatchSize, HaloWidth>(linear_idx);
            auto local_indices = remove_halo<Dim, HaloWidth>(coords_with_halo);

            auto cell_center =
                compute_cell_center<PatchSize, HaloWidth, Dim>(patch_id, local_indices);

            center_coord_patch[linear_idx] = cell_center;
            dof_patch[linear_idx]          = interpolator(cell_center, cell_size);
            eq->evaluate_flux(dof_patch[linear_idx], flux_patch[linear_idx]);
            size_patch[linear_idx] = { cell_size, cell_size };
        }
    }

    try
    {
        std::cout << "\n====================================\n";
        std::cout << "  Initializing DG Tree Printer (Advanced)\n";
        std::cout << "====================================\n\n";
        ndt::print::dg_tree_printer_advanced<Dim, Order, PatchSize, HaloWidth, DOFs>
            printer("dg_tree");
        std::cout << "DG tree printer created successfully\n";
        std::string time_extension = "_t" + std::to_string(timestep) + ".vtk";
        printer.template print<S1>(tree, time_extension);
        std::cout << "DG tree printer completed\n";

        // Print debug info for DOF inspection
        // advanced_printer.template print_debug_info<S1>(tree);

        std::vector<std::string> vtk_files;
        std::vector<double>      times;

        std::cout << "\n====================================\n";
        std::cout << "  Starting Time Integration\n";
        std::cout << "====================================\n\n";

        while (time < EndTime)
        {
            // Initialize halo cells with periodic boundary conditions
            tree.halo_exchange_update();
            // Apply time integrator to each patch in the tree
            for (std::size_t idx = 0; idx < tree.size(); ++idx)
            {
                auto& dof_patch    = tree.template get_patch<S1>(idx);
                auto& flux_patch   = tree.template get_patch<S2>(idx);
                auto& center_patch = tree.template get_patch<S3>(idx);

                // std::cout << "patch value " << dof_patch.data() << ": ";

                auto residual_callback = [&](patch_container_t&       patch_update,
                                             const patch_container_t& current_dofs,
                                             double                   t)
                {
                    Evaluator::evaluate(
                        *eq,
                        global_config.basis,
                        current_dofs,      // whole patch
                        flux_patch.data(), // whole flux patch
                        patch_update,      // whole RHS patch
                        center_patch.data(),
                        global_config.face_kernels,
                        t,
                        patch_layout_t{},
                        global_config,
                        0.01 // volume
                    );
                };

                integrator->step(residual_callback, dof_patch.data(), time, dt);
            }

            // Update halo cells with periodic boundary conditions
            tree.halo_exchange_update();
            if (timestep % 5 == 4)
            {
                // Print the VTU file
                time_extension = "_t" + std::to_string(timestep) + ".vtk";
                printer.template print<S1>(tree, time_extension);

                // Store the filename and time
                vtk_files.push_back(
                    "dg_tree_Order" + std::to_string(Order) + time_extension
                );
                times.push_back(time);
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