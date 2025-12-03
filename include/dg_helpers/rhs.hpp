#ifndef DG_HELPERS_RHS_HPP
#define DG_HELPERS_RHS_HPP

#include "basis.hpp"
#include "containers/container_manipulations.hpp"
#include "containers/container_operations.hpp"
#include "equations.hpp"
#include "globals.hpp"
#include "ndtree/ndtree.hpp"
#include "ndtree/neighbor.hpp"
#include "surface.hpp"
#include <functional>
#include <utility>
#include <variant>

namespace amr::rhs
{

// Helper function to dispatch to the correct compile-time direction
template <
    std::size_t Dim,
    typename KernelsType,
    typename DOFType,
    typename FluxType,
    typename SignType,
    std::size_t... Is>
auto dispatch_project_impl(
    std::index_sequence<Is...>,
    const KernelsType& kernels,
    const DOFType&     dof,
    const FluxType&    flux,
    SignType           sign,
    std::size_t        runtime_dim
)
{
    // Use a compile-time unrolled if-else chain
    std::decay_t<decltype(amr::surface::project_to_faces<KernelsType, 0>(
        kernels, dof, flux, sign
    ))>
        result;

    bool found = false;
    ((Is == runtime_dim ? (result = amr::surface::project_to_faces<KernelsType, Is>(
                               kernels, dof, flux, sign
                           ),
                           found = true)
                        : false) ||
     ...);

    return result;
}

template <
    std::size_t Dim,
    typename KernelsType,
    typename DOFType,
    typename FluxType,
    typename SignType>
auto dispatch_project_to_faces(
    const KernelsType& kernels,
    const DOFType&     dof,
    const FluxType&    flux,
    SignType           sign,
    std::size_t        runtime_dim
)
{
    return dispatch_project_impl<Dim>(
        std::make_index_sequence<Dim>{}, kernels, dof, flux, sign, runtime_dim
    );
}

/**
 * @brief Evaluate the RHS of the DG discretization
 *
 * Computes du/dt = -∇·F(u) for the entire patch.
 *
 * Template parameters:
 * - Order: DG basis order
 * - Dim: Spatial dimension
 * - PatchSize: Number of cells per patch dimension
 * - HaloWidth: Halo width of patches
 * - DOFs: Number of DOF components
 *
 * @param eq Equation instance (provides flux computation)
 * @param basis DG basis (provides quadrature points and weights)
 * @param dof_patch Current DOF values for entire patch (with halo)
 * @param flux_patch Storage for computed fluxes (with halo)
 * @param rhs_patch Output: RHS values for entire patch (interior only)
 * @param cell_size Physical size of each cell
 * @param kernels Pre-computed face kernels for efficiency
 */
template <
    std::size_t Order,
    std::size_t Dim,
    std::size_t PatchSize,
    std::size_t HaloWidth,
    std::size_t DOFs,
    typename EquationType,
    typename BasisType,
    typename DOFTensorType,
    typename FluxVectorType,
    typename KernelsType,
    typename PatchLayoutType>
inline void evaluate_rhs(
    const EquationType&               eq,
    [[maybe_unused]] const BasisType& basis,
    DOFTensorType&                    dof_patch,
    FluxVectorType&                   flux_patch,
    DOFTensorType&                    rhs_patch,
    [[maybe_unused]] double           cell_size,
    const KernelsType&                kernels,
    [[maybe_unused]] double&          t,
    const PatchLayoutType&            patch_layout_t
)
{
    using direction_t     = amr::ndt::neighbors::direction<Dim>;
    using padded_layout_t = typename PatchLayoutType::padded_layout_t;
    using flux_vector_t   = std::decay_t<decltype(flux_patch[0])>;
    using size_type       = typename flux_vector_t::size_type;

    // Initialize RHS buffer to zero for all cells (interior and halo)
    for (std::size_t linear_idx = 0; linear_idx < patch_layout_t.flat_size();
         ++linear_idx)
    {
        rhs_patch[linear_idx] = std::decay_t<decltype(rhs_patch[0])>{};
    }

    // Loop over each cell in the patch (including halos)
    for (std::size_t linear_idx = 0; linear_idx < patch_layout_t.flat_size();
         ++linear_idx)
    {
        if (amr::ndt::utils::patches::is_halo_cell<PatchLayoutType>(linear_idx))
        {
            // std::cout << "  [HALO] Skipping halo cell at linear_idx=" << linear_idx
            //           << "\n";
            continue;
        }

        // std::cout << "  [INTERIOR] Processing interior cell at linear_idx=" <<
        // linear_idx;

        // Get cell coordinates for display
        // auto cell_coords = padded_layout_t::multi_index(
        //     static_cast<typename padded_layout_t::index_t>(linear_idx)
        // );
        // std::cout << " coords=[" << cell_coords[0];
        // for (std::size_t i = 1; i < Dim; ++i)
        //     std::cout << "," << cell_coords[i];
        // std::cout << "]\n";

        // Compute flux for current cell
        eq.evaluate_flux(dof_patch[linear_idx], flux_patch[linear_idx]);

        // Loop over neighbor directions
        for (auto d = direction_t::first(); d != direction_t::sentinel(); d.advance())
        {
            // Use padded_layout_t for coordinate conversion (12×12)
            auto neighbor_coords = padded_layout_t::multi_index(
                static_cast<typename padded_layout_t::index_t>(linear_idx)
            );

            const std::size_t dim  = d.dimension();
            const int         sign = direction_t::is_negative(d) ? 0 : 1;

            neighbor_coords[dim] += (direction_t::is_negative(d) ? -1 : 1);

            // Convert back to linear index using padded_layout_t
            auto neighbor_linear_idx = padded_layout_t::linear_index(neighbor_coords);

            // Bounds check using padded layout size
            if (neighbor_linear_idx >= patch_layout_t.flat_size()) continue;

            const auto dim_idx = static_cast<size_type>(dim);

            // std::string direction_str = (direction_t::is_negative(d) ? "-" : "+");
            // std::cout << "    [PROJECTION] dim=" << dim << ", direction=" <<
            // direction_str
            //           << ", neighbor_coords=[" << neighbor_coords[0];
            // for (std::size_t i = 1; i < Dim; ++i)
            //     std::cout << "," << neighbor_coords[i];
            // std::cout << "], neighbor_idx=" << neighbor_linear_idx << "\n"; // Project
            // to faces using dimension dispatch flux_patch[linear_idx] is S2::type =
            // static_vector<dof_t, Dim> flux_patch[linear_idx][dim_idx] is dof_t = the
            // flux tensor for dimension 'dim'

            auto cell_data = dispatch_project_to_faces<Dim>(
                kernels, dof_patch[linear_idx], flux_patch[linear_idx][dim_idx], sign, dim
            );

            auto neighbor_data = dispatch_project_to_faces<Dim>(
                kernels,
                dof_patch[neighbor_linear_idx],
                flux_patch[neighbor_linear_idx][dim_idx],
                !sign,
                dim
            );
            [[maybe_unused]]
            double curr_eigenval{};
            auto   face_du = amr::surface::evaluate_face_integral(
                curr_eigenval,
                eq,
                kernels,
                std::get<0>(cell_data),
                std::get<0>(neighbor_data),
                std::get<1>(cell_data),
                std::get<1>(neighbor_data),
                dim,
                sign,
                0.1 // edge surface
            );
            // TODO: Compute numerical flux and update RHS
            // eq.numerical_flux(cell_data, neighbor_data, rhs_patch[linear_idx], dim,
            // sign);

            rhs_patch[linear_idx] = rhs_patch[linear_idx] + face_du;
        }
    }
    dof_patch = rhs_patch;
}

} // namespace amr::rhs

#endif // DG_HELPERS_RHS_HPP