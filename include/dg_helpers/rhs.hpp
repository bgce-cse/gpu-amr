#ifndef DG_HELPERS_RHS_HPP
#define DG_HELPERS_RHS_HPP

#include "basis/basis.hpp"
#include "containers/container_manipulations.hpp"
#include "containers/container_operations.hpp"
#include "equations/equations.hpp"
#include "globals/global_config.hpp"
#include "globals/globals.hpp"
#include "ndtree/ndtree.hpp"
#include "ndtree/neighbor.hpp"
#include "surface.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <utility>
#include <variant>

namespace amr::rhs
{

// Helper function to dispatch to the correct compile-time direction

/**
 * @brief RHS Evaluator for DG discretization
 *
 * Statically caches mass tensors and compile-time constants to avoid
 * repeated allocation and computation across multiple RHS evaluations.
 */
template <typename global_t, typename Policy>
struct RHSEvaluator
{
    using eq        = typename global_t::EquationImpl;
    using surface_t = amr::surface::Surface<global_t, Policy>;

    template <
        std::size_t Dim,
        typename DOFType,
        typename FluxType,
        typename SignType,
        std::size_t... Is>
    static auto dispatch_project_impl(
        std::index_sequence<Is...>,
        const DOFType&  dof,
        const FluxType& flux,
        SignType        sign,
        std::size_t     runtime_dim
    )
    {
        // Use a compile-time unrolled if-else chain
        // Note: project_to_faces requires <KernelsType, Direction, Tensor>
        // We instantiate with void as KernelsType (unused), Direction as Is, Tensor as
        // DOFType
        std::decay_t<
            decltype(surface_t::template project_to_faces<0, DOFType>(dof, flux, sign))>
            result;

        bool found = false;
        ((Is == runtime_dim
              ? (result =
                     surface_t::template project_to_faces<Is, DOFType>(dof, flux, sign),
                 found = true)
              : false) ||
         ...);

        return result;
    }

    template <std::size_t Dim, typename DOFType, typename FluxType, typename SignType>
    static auto dispatch_project_to_faces(
        const DOFType&  dof,
        const FluxType& flux,
        SignType        sign,
        std::size_t     runtime_dim
    )
    {
        return dispatch_project_impl<Dim>(
            std::make_index_sequence<Dim>{}, dof, flux, sign, runtime_dim
        );
    }

    template <
        typename DOFTensorType,
        typename FluxVectorType,
        typename PatchLayoutType,
        typename CenterType>
    inline static double evaluate(
        const DOFTensorType&               dof_patch,    // const: read-only
        FluxVectorType&                    flux_patch,   // written inside
        DOFTensorType&                     patch_update, // output
        [[maybe_unused]] const CenterType& cell_center,  // read-only
        [[maybe_unused]] double const&     dt,
        const PatchLayoutType&             patch_layout_t,
        [[maybe_unused]] const double&     volume,
        const double&                      surface
    )
    {
        using direction_t     = amr::ndt::neighbors::direction<Policy::Dim>;
        using padded_layout_t = typename PatchLayoutType::padded_layout_t;
        using flux_vector_t   = std::decay_t<decltype(flux_patch[0])>;
        using size_type       = typename flux_vector_t::size_type;
        using size_t          = typename PatchLayoutType::size_type;

        double maxeigenval = -std::numeric_limits<double>::infinity();

        // Loop over each cell in the patch (including halos)
        for (size_t linear_idx = 0; linear_idx < patch_layout_t.flat_size(); ++linear_idx)
        {
            auto idx = static_cast<int>(linear_idx);

            if (amr::ndt::utils::patches::is_halo_cell<PatchLayoutType>(idx))
            {
                continue;
            }
            // compute flux from current DOFs
            flux_patch[idx] = eq::evaluate_flux(dof_patch[idx]);

            for (auto d = direction_t::first(); d != direction_t::sentinel(); d.advance())
            {
                // Convert current linear index to multi-index using padded_layout_t
                auto current_coords = padded_layout_t::multi_index(
                    static_cast<typename padded_layout_t::index_t>(idx)
                );

                // Natural neighbor in dimension d
                auto dim_idx  = static_cast<int>(d.dimension());
                bool is_neg   = direction_t::is_negative(d);
                int  sign     = is_neg ? -1 : 1;
                int  sign_idx = is_neg ? 0 : 1;

                // Compute neighbor
                auto neighbor_coords = current_coords;
                neighbor_coords[dim_idx] += sign;

                auto neighbor_linear_idx = padded_layout_t::linear_index(neighbor_coords);

                // Bounds check using padded layout size
                if (neighbor_linear_idx >= padded_layout_t::flat_size())
                {
                    std::cout << "[BOUNDS] OUT OF BOUNDS\n";
                    continue;
                }

                auto actual_dim    = Policy::Dim - dim_idx - 1;
                auto actual_dim_sz = static_cast<size_type>(actual_dim);

                // project to face (caller supplies face_kernels via global_t)
                auto cell_data = dispatch_project_to_faces<Policy::Dim>(
                    dof_patch[idx], flux_patch[idx][actual_dim_sz], sign_idx, actual_dim
                );

                auto neighbor_data = dispatch_project_to_faces<Policy::Dim>(
                    dof_patch[neighbor_linear_idx],
                    flux_patch[neighbor_linear_idx][actual_dim_sz],
                    1 - sign_idx,
                    actual_dim
                );

                // Compute maximum eigenvalue for Rusanov flux
                // Use the maximum eigenvalue from the original DOF tensors at the cell
                // center
                double eigenval_curr = eq::max_eigenvalue(
                    dof_patch[idx], static_cast<unsigned int>(actual_dim)
                );
                double eigenval_neigh = eq::max_eigenvalue(
                    dof_patch[neighbor_linear_idx], static_cast<unsigned int>(actual_dim)
                );
                double curr_eigenval = std::max(eigenval_curr, eigenval_neigh);

                // Track maximum eigenvalue across all faces
                maxeigenval = std::max(maxeigenval, curr_eigenval);

                auto face_du = surface_t::evaluate_face_integral(
                    curr_eigenval,
                    std::get<0>(cell_data),
                    std::get<0>(neighbor_data),
                    std::get<1>(cell_data),
                    std::get<1>(neighbor_data),
                    actual_dim,
                    sign,
                    sign_idx,
                    surface
                );

                patch_update[idx] = patch_update[idx] - face_du;
            }

            // apply inverse mass tensor (per-cell normalization: volume *
            // reference_massmatrix inverse) In the Julia reference, this is:
            // inv_massmatrix = inv(volume(cell) * reference_massmatrix) Which means: du =
            // (1 / (volume * reference_mass)) * du = (1/volume) * inv(reference_mass) *
            // du
            patch_update[idx] = amr::containers::algorithms::tensor::tensor_dot(
                patch_update[idx], global_t::inv_volume_mass / volume
            );
            // std::cout << "inverse volume" << global_t::inv_volume_mass / volume <<
            // "\n";
        }

        return maxeigenval;
    }
};

} // namespace amr::rhs

#endif // DG_HELPERS_RHS_HPP