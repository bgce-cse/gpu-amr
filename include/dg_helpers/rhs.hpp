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
    const DOFTensorType&              dof_patch,
    FluxVectorType&                   flux_patch,
    [[maybe_unused]] DOFTensorType&   rhs_patch,
    [[maybe_unused]] double           cell_size,
    const KernelsType&                kernels,
    [[maybe_unused]] double&          t,
    const PatchLayoutType&            patch_layout_t
)
{
    using direction_t   = amr::ndt::neighbors::direction<Dim>;
    using data_layout_t = typename PatchLayoutType::data_layout_t;
    using flux_vector_t = std::decay_t<decltype(flux_patch[0])>;
    using size_type     = typename flux_vector_t::size_type;

    // Loop over each cell in the patch
    for (std::size_t linear_idx = 0; linear_idx < patch_layout_t.flat_size();
         ++linear_idx)
    {
        // Compute flux for current cell
        eq.evaluate_flux(dof_patch[linear_idx], flux_patch[linear_idx]);

        // Loop over neighbor directions
        for (auto d = direction_t::first(); d != direction_t::sentinel(); d.advance())
        {
            auto neighbor_coords = data_layout_t::multi_index(
                static_cast<typename data_layout_t::index_t>(linear_idx)
            );
            const std::size_t dim  = d.dimension();
            const int         sign = direction_t::is_negative(d) ? 0 : 1;

            neighbor_coords[dim] += (direction_t::is_negative(d) ? -1 : 1);

            // Check if neighbor is within patch interior
            bool in_bounds = true;
            for (std::size_t i = 0; i < Dim; ++i)
            {
                if (neighbor_coords[i] < 0 ||
                    neighbor_coords[i] >= static_cast<int>(PatchSize))
                {
                    in_bounds = false;
                    break;
                }
            }

            if (!in_bounds) continue;

            auto neighbor_linear_idx = data_layout_t::linear_index(neighbor_coords);

            // Cast dim to the size_type of the flux vector to avoid conversion warnings
            const auto dim_idx = static_cast<size_type>(dim);

            // Project to faces using dimension dispatch
            // flux_patch[linear_idx] is S2::type = static_vector<dof_t, Dim>
            // flux_patch[linear_idx][dim_idx] is dof_t = the flux tensor for dimension
            // 'dim'
            [[maybe_unused]]
            auto cell_data = dispatch_project_to_faces<Dim>(
                kernels, dof_patch[linear_idx], flux_patch[linear_idx][dim_idx], sign, dim
            );

            [[maybe_unused]]
            auto neighbor_data = dispatch_project_to_faces<Dim>(
                kernels,
                dof_patch[neighbor_linear_idx],
                flux_patch[neighbor_linear_idx][dim_idx],
                !sign,
                dim
            );

            // TODO: Compute numerical flux and update RHS
            // eq.numerical_flux(cell_data, neighbor_data, rhs_patch[linear_idx], dim,
            // sign);
        }
    }
}

} // namespace amr::rhs

#endif // DG_HELPERS_RHS_HPP