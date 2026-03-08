#ifndef DG_HELPERS_RHS_HPP
#define DG_HELPERS_RHS_HPP

#include "basis/basis.hpp"
#include "containers/container_manipulations.hpp"
#include "containers/container_operations.hpp"
#include "dg_helpers/tree_builder.hpp"
#include "equations/equations.hpp"
#include "globals/global_config.hpp"
#include "globals/globals.hpp"
#include "ndtree/ndtree.hpp"
#include "ndtree/neighbor.hpp"
#include "surface.hpp"
#include "volume.hpp"
#include <algorithm>
#include <cmath>
#include <utility>

namespace amr::rhs
{

namespace tensor_ops = amr::containers::algorithms::tensor;
namespace patch_util = amr::ndt::utils::patches;

/**
 * @brief RHS evaluator for the DG spatial discretization.
 *
 * Computes  dU/dt = − (surface terms) + (volume terms)
 * for every interior cell of a patch.  Dimension-generic.
 *
 * @tparam global_t  Fully-assembled GlobalConfig type.
 * @tparam Policy    Compile-time configuration policy.
 */
template <typename global_t, typename Policy>
struct RHSEvaluator
{
    // -----------------------------------------------------------------
    //  Type aliases
    // -----------------------------------------------------------------
    using eq        = typename global_t::EquationImpl;
    using surface_t = amr::surface::Surface<global_t, Policy>;
    using volume_t  = amr::volume::VolumeEvaluator<global_t, Policy>;
    using dg_tree = amr::dg_tree::TreeBuilder<global_t, amr::config::GlobalConfigPolicy>;
    using patch_layout_t  = typename dg_tree::patch_layout_t;
    using direction_t     = amr::ndt::neighbors::direction<Policy::Dim>;
    using padded_layout_t = typename patch_layout_t::padded_layout_t;
    using flux_vector_t   = typename dg_tree::S1::type;
    using size_type       = typename flux_vector_t::size_type;
    using size_t          = typename patch_layout_t::size_type;

    // -----------------------------------------------------------------
    //  Direction dispatch
    //
    //  project_to_faces needs a compile-time Direction template arg;
    //  these helpers map a runtime dimension index to the correct
    //  Compile-time direction processing: processes one surface
    //  direction (dimension + sign) with all parameters resolved
    //  at compile time, eliminating runtime dispatch overhead.
    // -----------------------------------------------------------------
    template <
        std::size_t DimIdx,
        bool        IsNeg,
        typename DOFTensorType,
        typename FluxVectorType>
    static void process_surface_direction(
        const DOFTensorType&  dof_patch,
        const FluxVectorType& flux_patch,
        DOFTensorType&        patch_update,
        const double&         surface,
        double&               max_eigenval,
        size_t                idx,
        auto const&           current_coords
    )
    {
        static constexpr int sign       = IsNeg ? -1 : 1;
        static constexpr int sign_idx   = IsNeg ? 0 : 1;
        static constexpr int actual_dim = static_cast<int>(Policy::Dim - DimIdx - 1);

        auto neighbor_coords = current_coords;
        neighbor_coords[DimIdx] += sign;
        auto neighbor_linear_idx = padded_layout_t::linear_index(neighbor_coords);

        auto [dofs_face, flux_face] = surface_t::template project_to_faces<actual_dim>(
            dof_patch[idx], flux_patch[idx][actual_dim], sign_idx
        );
        auto [dofs_face_neigh, flux_face_neigh] =
            surface_t::template project_to_faces<actual_dim>(
                dof_patch[neighbor_linear_idx],
                flux_patch[neighbor_linear_idx][actual_dim],
                1 - sign_idx
            );

        size_t dim   = actual_dim;
        max_eigenval = maxeigenvalue(
            max_eigenval, dof_patch[idx], dof_patch[neighbor_linear_idx], dim
        );

        auto face_du = surface_t::template evaluate_face_integral<actual_dim>(
            dofs_face,
            dofs_face_neigh,
            flux_face,
            flux_face_neigh,
            sign,
            sign_idx,
            surface,
            max_eigenval
        );

        patch_update[idx] -= face_du;
    }

    // -----------------------------------------------------------------
    //  Max eigenvalue across two neighbouring cells
    // -----------------------------------------------------------------
    static double
        maxeigenvalue(auto& max_eigenval, auto& dof_cell, auto& dof_neigh, auto& dim)
    {
        double curr_eigenval = std::max(
            eq::max_eigenvalue(dof_cell, dim), eq::max_eigenvalue(dof_neigh, dim)
        );
        return std::max(max_eigenval, curr_eigenval);
    }

    // -----------------------------------------------------------------
    //  Main entry point: evaluate the full RHS for one patch
    // -----------------------------------------------------------------
    template <typename DOFTensorType, typename FluxVectorType>
    inline static void evaluate(
        const DOFTensorType&           dof_patch,
        FluxVectorType&                flux_patch,
        DOFTensorType&                 patch_update,
        [[maybe_unused]] double const& dt,
        const double&                  volume,
        const double&                  surface,
        double&                        max_eigenval,
        const double&                  inverse_jacobian
    )
    {
        compute_fluxes_and_volume(
            dof_patch, flux_patch, patch_update, volume, inverse_jacobian
        );
        accumulate_surface_terms(
            dof_patch, flux_patch, patch_update, surface, max_eigenval
        );
        apply_inverse_mass(patch_update, volume);
    }

private:
    // -----------------------------------------------------------------
    //  Step 1: Evaluate physical fluxes and volume integrals
    // -----------------------------------------------------------------
    template <typename DOFTensorType, typename FluxVectorType>
    static void compute_fluxes_and_volume(
        const DOFTensorType& dof_patch,
        FluxVectorType&      flux_patch,
        DOFTensorType&       patch_update,
        const double&        volume,
        const double&        inverse_jacobian
    )
    {
        for (size_t linear_idx = 0; linear_idx < patch_layout_t::flat_size();
             ++linear_idx)
        {
            flux_patch[linear_idx] = eq::evaluate_flux(dof_patch[linear_idx]);

            if (Policy::Order > 1 &&
                !patch_util::is_halo_cell<patch_layout_t>(linear_idx))
            {
                volume_t::evaluate_volume_integral(
                    patch_update[linear_idx],
                    flux_patch[linear_idx],
                    volume,
                    inverse_jacobian
                );
            }
        }
    }

    // -----------------------------------------------------------------
    //  Step 2: Loop over interior cells × directions, accumulate
    //          surface flux contributions into patch_update
    // -----------------------------------------------------------------
    template <typename DOFTensorType, typename FluxVectorType>
    static void accumulate_surface_terms(
        const DOFTensorType&  dof_patch,
        const FluxVectorType& flux_patch,
        DOFTensorType&        patch_update,
        const double&         surface,
        double&               max_eigenval
    )
    {
        for (size_t idx = 0; idx < patch_layout_t::flat_size(); ++idx)
        {
            if (patch_util::is_halo_cell<patch_layout_t>(idx)) continue;

            auto current_coords = padded_layout_t::multi_index(
                static_cast<typename padded_layout_t::index_t>(idx)
            );

            // Compile-time unroll over all 2*Dim face directions
            [&]<std::size_t... Ds>(std::index_sequence<Ds...>)
            {
                ((process_surface_direction<Ds, true>(
                      dof_patch,
                      flux_patch,
                      patch_update,
                      surface,
                      max_eigenval,
                      idx,
                      current_coords
                  ),
                  process_surface_direction<Ds, false>(
                      dof_patch,
                      flux_patch,
                      patch_update,
                      surface,
                      max_eigenval,
                      idx,
                      current_coords
                  )),
                 ...);
            }(std::make_index_sequence<Policy::Dim>{});
        }
    }

    // -----------------------------------------------------------------
    //  Step 3: Apply the inverse volume mass matrix to every
    //          interior cell
    // -----------------------------------------------------------------
    template <typename DOFTensorType>
    static void apply_inverse_mass(DOFTensorType& patch_update, const double& volume)
    {
        for (size_t idx = 0; idx < patch_layout_t::flat_size(); ++idx)
        {
            if (patch_util::is_halo_cell<patch_layout_t>(idx)) continue;

            patch_update[idx] *= global_t::inv_volume_mass / volume;
        }
    }
};

} // namespace amr::rhs

#endif // DG_HELPERS_RHS_HPP