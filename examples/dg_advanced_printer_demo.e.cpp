/**
 * @file dg_advanced_printer_demo.e.cpp
 * @brief Example demonstrating the advanced DG tree printer
 *
 * This example shows how to use the dg_tree_printer_advanced template
 * to output DG AMR tree data to VTK format with:
 * - Arbitrary physical dimensions (2D, 3D, etc.)
 * - Arbitrary DG polynomial orders
 * - Proper DOF tensor extraction at Gauss-Legendre quadrature points
 * - Per-component scalar field output
 *
 * The printer works with any DG tree structure and extracts:
 * - Cell coordinates in global space
 * - DOF values from S1 tensors at each quadrature point
 * - All components of the DOF vector at each point
 *
 * Usage:
 *   cd build
 *   cmake ../examples
 *   make dg_advanced_printer_demo
 *   ./bin/Debug/dg_advanced_printer_demo
 *   # Output appears in: vtk_output/dg_tree_advanced_*.vtk
 */

#include "containers/static_layout.hpp"
#include "containers/static_shape.hpp"
#include "containers/static_vector.hpp"
#include "dg_helpers/basis.hpp"
#include "dg_helpers/equations.hpp"
#include "dg_helpers/equations/advection.hpp"
#include "dg_helpers/globals.hpp"
#include "generated_config.hpp"
#include "morton/morton_id.hpp"
#include "ndtree/ndhierarchy.hpp"
#include "ndtree/ndtree.hpp"
#include "ndtree/patch_layout.hpp"
#include "ndtree/print_dg_tree_advanced.hpp"
#include "utility/random.hpp"
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <tuple>

using namespace amr::equations;
using namespace amr::Basis;
using namespace amr::containers;
using namespace amr::config;

/**
 * @brief DG cell structure with S1: DOF tensor at Gauss-Legendre points
 *
 * S1 structure:
 * - Type: hypercube tensor with shape (Order+1, Order+1, ..., Order+1) in Dim
 * - At each tensor entry: static_vector<double, DOFs> containing DOF values
 */
struct S1
{
    using dof_value_t = amr::containers::static_vector<double, DOFs>;
    using type        = typename amr::containers::utils::types::tensor::
        hypercube_t<dof_value_t, Order, Dim>;

    static constexpr auto index() noexcept -> std::size_t
    {
        return 0;
    }

    type value;
};

/**
 * @brief Flux tensor structure (used in real computations)
 */
struct S2
{
    using dof_value_t = amr::containers::static_vector<double, DOFs>;
    using dof_t       = typename amr::containers::utils::types::tensor::
        hypercube_t<dof_value_t, Order, Dim>;
    using type = amr::containers::static_vector<dof_t, Dim>;

    static constexpr auto index() noexcept -> std::size_t
    {
        return 1;
    }

    type value;
};

/**
 * @brief Cell center coordinates
 */
struct S3
{
    using type = typename amr::containers::static_vector<double, Dim>;

    static constexpr auto index() noexcept -> std::size_t
    {
        return 2;
    }

    type value;
};

/**
 * @brief Marker cell type for tree AMR with DG data
 *
 * Contains:
 * - S1: DOF tensor at quadrature points (main data for printer)
 * - S2: Flux vectors
 * - S3: Cell center coordinates
 */
// hi
struct MarkerCell
{
    using deconstructed_types_map_t = std::tuple<S1, S2, S3>;

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
    return os << "S1(DOF Tensor), S2(Flux), S3(Center)";
}

/**
 * @brief Main demonstration of advanced DG printer
 *
 * Creates a simple DG AMR tree with synthetic initial conditions
 * and outputs to VTK format using the advanced printer.
 */
int main()
{
    std::cout << "====================================\n";
    std::cout << "  Advanced DG Printer Demo\n";
    std::cout << "====================================\n\n";

    // Configuration
    std::cout << "Configuration:\n";
    std::cout << "  Physical Dimension: " << Dim << "D\n";
    std::cout << "  DG Polynomial Order: " << Order << "\n";
    std::cout << "  DOF Components: " << DOFs << "\n";
    std::cout << "  Equation: " << Equation << "\n\n";

    // Check configuration compatibility
    if (Dim != 2)
    {
        std::cerr << "Error: This example supports 2D only (Dim must be 2)\n";
        return 1;
    }

    // Initialize basis
    Basis<Order, Dim> basis(0.0, GridSize);
    auto              eq = make_configured_equation();

    // Create interpolator for initial DOF values
    auto interpolator = amr::equations::InitialDOFInterpolator(*eq, basis);

    // Setup tree mesh
    constexpr std::size_t PatchSize = 4;
    constexpr std::size_t HaloWidth = 1;
    constexpr std::size_t MaxDepth  = 2;

    using shape_t        = static_shape<PatchSize, PatchSize>;
    using layout_t       = static_layout<shape_t>;
    using patch_index_t  = amr::ndt::morton::morton_id<MaxDepth, Dim>;
    using patch_layout_t = amr::ndt::patches::patch_layout<layout_t, HaloWidth>;
    using tree_t = amr::ndt::tree::ndtree<MarkerCell, patch_index_t, patch_layout_t>;

    tree_t tree(10000);

    std::cout << "Tree Configuration:\n";
    std::cout << "  Max Depth: " << MaxDepth << "\n";
    std::cout << "  Patch Size: " << PatchSize << "x" << PatchSize << "\n";
    std::cout << "  Halo Width: " << HaloWidth << "\n";
    std::cout << "  Patches: " << tree.size() << "\n\n";

    // Initialize DG patches with synthetic data
    std::cout << "Initializing DG patches...\n";

    for (std::size_t idx = 0; idx < tree.size(); ++idx)
    {
        auto& dof_patch          = tree.template get_patch<S1>(idx);
        auto& center_coord_patch = tree.template get_patch<S3>(idx);

        auto patch_id                    = patch_index_t(idx);
        auto [patch_coords, patch_level] = patch_index_t::decode(patch_id.id());
        double patch_level_size          = 1.0 / static_cast<double>(1u << patch_level);
        double cell_size = patch_level_size / static_cast<double>(PatchSize);

        for (std::size_t linear_idx = 0; linear_idx != patch_layout_t::flat_size();
             ++linear_idx)
        {
            // Skip halo cells
            if (amr::ndt::utils::patches::is_halo_cell<patch_layout_t>(linear_idx))
                continue;

            // Compute local cell coordinates from linear index
            std::size_t local_x_with_halo = linear_idx % (PatchSize + 2 * HaloWidth);
            std::size_t local_y_with_halo = linear_idx / (PatchSize + 2 * HaloWidth);
            std::size_t local_x           = local_x_with_halo - HaloWidth;
            std::size_t local_y           = local_y_with_halo - HaloWidth;

            amr::containers::static_vector<double, Dim> cell_center_coords;
            // Compute global cell center based on patch and local cell position
            cell_center_coords[0] =
                (static_cast<double>(patch_coords[0]) * static_cast<double>(PatchSize) +
                 static_cast<double>(local_x) + 0.5) *
                cell_size;
            cell_center_coords[1] =
                (static_cast<double>(patch_coords[1]) * static_cast<double>(PatchSize) +
                 static_cast<double>(local_y) + 0.5) *
                cell_size;

            center_coord_patch[linear_idx] = cell_center_coords;

            // Initialize DOF tensor with interpolated initial conditions
            dof_patch[linear_idx] = interpolator(cell_center_coords, cell_size);
        }
    }

    std::cout << "Initialization complete.\n\n";

    // ========================================================================
    // Use the ADVANCED printer - supports arbitrary Dim and Order
    // ========================================================================
    std::cout << "====================================\n";
    std::cout << "  Printing with Advanced Printer\n";
    std::cout << "====================================\n\n";

    // Create printer with template parameters:
    // <Dim, Order, PatchSize, HaloWidth, NumDOFs>
    ndt::print::dg_tree_printer_advanced<Dim, Order, PatchSize, HaloWidth, DOFs>
        advanced_printer("dg_tree_advanced_timestep");

    // Print to VTK file
    advanced_printer.template print<S1>(tree, "_0.vtk");

    std::cout << "VTK file written to: vtk_output/dg_tree_advanced_timestep_0.vtk\n\n";

    // ========================================================================
    // Print sample DOF values to verify extraction
    // ========================================================================
    std::cout << "====================================\n";
    std::cout << "  Sample DOF Values at Quadrature Points\n";
    std::cout << "====================================\n\n";

    for (std::size_t patch_idx = 0; patch_idx < std::min(tree.size(), std::size_t(1));
         ++patch_idx)
    {
        const auto& dof_patch = tree.template get_patch<S1>(patch_idx);
        std::cout << "Patch " << patch_idx << ":\n";

        for (std::size_t local_y = 0; local_y < std::min(PatchSize, std::size_t(2));
             ++local_y)
        {
            for (std::size_t local_x = 0; local_x < std::min(PatchSize, std::size_t(2));
                 ++local_x)
            {
                std::size_t linear_idx =
                    (local_y + HaloWidth) * (PatchSize + 2 * HaloWidth) +
                    (local_x + HaloWidth);

                // Get the DOF tensor for this cell
                const auto& cell_dofs = dof_patch[linear_idx];

                std::cout << "  Cell [" << local_x << "," << local_y
                          << "] Quadrature Points:\n";

                // For Order=2 (3 points per dimension in 2D = 9 points total)
                for (std::size_t j = 0; j <= Order && j < 2; ++j)
                {
                    for (std::size_t i = 0; i <= Order && i < 2; ++i)
                    {
                        const auto& dof_vec = cell_dofs[i, j];
                        std::cout << "    Q[" << i << "," << j << "]: ";
                        for (unsigned int d = 0; d < DOFs; ++d)
                        {
                            std::cout << std::scientific << std::setprecision(3)
                                      << dof_vec[d] << " ";
                        }
                        std::cout << "\n";
                    }
                }
            }
        }
        std::cout << "\n";
    }

    std::cout << "====================================\n";
    std::cout << "  Demo Complete\n";
    std::cout << "====================================\n";

    return 0;
}
