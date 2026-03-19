#pragma once

#include "containers/container_algorithms.hpp"
#include "containers/static_vector.hpp"
#include "config/generated_config.hpp"
#include "globals/global_config.hpp"
#include "globals/globals.hpp"
#include <tuple>

namespace amr::surface
{

namespace tensor_ops = amr::containers::algorithms::tensor;

/**
 * @brief Surface integral evaluator for the DG discretization.
 *
 * Handles face projection, Rusanov numerical flux computation, and
 * the surface-integral contribution to the RHS.  Fully dimension-generic.
 *
 * @tparam global_t  Fully-assembled GlobalConfig type.
 * @tparam Policy    Compile-time configuration policy.
 */
template <typename global_t, typename Policy>
struct Surface
{
    // -----------------------------------------------------------------
    //  Cached compile-time data
    // -----------------------------------------------------------------
    using global_config                 = global_t::GlobalConfig;
    static constexpr auto& weights_vec  = global_config::quad_weights;
    static constexpr auto& surface_mass = global_config::surface_mass;
    static constexpr auto& kernels      = global_config::face_kernels;

    // -----------------------------------------------------------------
    //  Rusanov (local Lax-Friedrichs) numerical flux
    //  F* = 0.5 * sign * (F_L + F_R) + 0.5 * λ_max * (u_L − u_R),
    //  scaled by the face area.
    // -----------------------------------------------------------------
    template <typename Tensor>
    static void rusanov(
        const Tensor&                dofs_face,
        const Tensor&                dofs_face_neigh,
        const Tensor&                flux_face,
        const Tensor&                flux_face_neigh,
        double                       surface,
        [[maybe_unused]] std::size_t direction,
        int                          sign,
        double                       max_eigenval,
        Tensor&                      numericalflux
    )
    {
        numericalflux = (sign * (flux_face + flux_face_neigh) * 0.5 +
                         (dofs_face - dofs_face_neigh) * (0.5 * max_eigenval)) *
                        surface;
    }

    // -----------------------------------------------------------------
    //  Face projection: contract the volume tensor with the face
    //  interpolation kernel along compile-time Direction.
    // -----------------------------------------------------------------
    template <int Direction, typename Tensor>
    static auto project_to_faces(
        const Tensor&        dofs,
        const Tensor&        flux,
        [[maybe_unused]] int sign
    )
    {
        auto dofs_face = tensor_ops::template contract<Direction>(dofs, kernels[sign]);
        auto flux_face = tensor_ops::template contract<Direction>(flux, kernels[sign]);
        return std::make_pair(dofs_face, flux_face);
    }

    // -----------------------------------------------------------------
    //  Full surface integral for one face.
    //
    //  1. Compute Rusanov numerical flux.
    //  2. Weight by the surface mass matrix.
    //  3. Lift back to the volume via a tensor product with the
    //     face kernel along the appropriate dimension.
    // -----------------------------------------------------------------
    template <typename Tensor>
    static auto evaluate_face_integral(
        const Tensor& dofs_face,
        const Tensor& dofs_face_neigh,
        const Tensor& flux_face,
        const Tensor& flux_face_neigh,
        std::size_t   direction,
        int           sign,
        int           sign_idx,
        double        surface,
        double&       max_eigenval
    )
    {
        Tensor numericalflux{};
        rusanov<Tensor>(
            dofs_face,
            dofs_face_neigh,
            flux_face,
            flux_face_neigh,
            surface,
            direction,
            sign,
            max_eigenval,
            numericalflux
        );

        auto kernel_vec    = kernels[sign_idx];
        auto weighted_flux = tensor_ops::tensor_dot(numericalflux, surface_mass);

        // Lift into the correct volume dimension via tensor product
        if (direction == 1) return tensor_ops::tensor_product(weighted_flux, kernel_vec);
        return tensor_ops::tensor_product(kernel_vec, weighted_flux);
    }
};

} // namespace amr::surface
