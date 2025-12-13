#pragma once

#include "containers/container_algorithms.hpp"
#include "containers/static_vector.hpp"
#include "generated_config.hpp"
#include "globals/global_config.hpp"
#include "globals/globals.hpp"
#include <cassert>
#include <tuple>

namespace amr::surface
{

template <typename global_t, typename Policy>
struct Surface
{
    using global_config                 = global_t::GlobalConfig;
    static constexpr auto& weights_vec  = global_config::quad_weights;
    static constexpr auto& surface_mass = global_config::surface_mass;
    static constexpr auto& kernels      = global_config::face_kernels;

    template <typename Tensor>
    static double rusanov(
        const Tensor&                dofs_face,
        const Tensor&                dofs_face_neigh,
        const Tensor&                flux_face,
        const Tensor&                flux_face_neigh,
        double                       surface,
        [[maybe_unused]] std::size_t direction,
        int                          sign,
        double                       maxeigenval,
        Tensor&                      numericalflux
    )
    {
        // Rusanov flux: F* = (F_L + F_R)/2 + lambda/2 * (U_L - U_R)
        // sign accounts for the outward normal direction for the flux average term
        // The dissipation term uses the jump without sign, as it's direction-independent
        numericalflux = (sign * (flux_face + flux_face_neigh) * 0.5 +
                         (dofs_face - dofs_face_neigh) * (0.5 * maxeigenval)) *
                        surface;
        // std::cout << "sign: " << sign << " numerical flux: " << numericalflux << "\n"
        //           << "dofs_face: " << dofs_face << " flux_face: " << flux_face << "\n"
        //           << "dofs_neigh: " << dofs_face_neigh
        //           << " flux_face_neigh: " << flux_face_neigh << "\n"
        //           << "surface: " << surface << " maxeigenval: " << maxeigenval << "\n";
        return maxeigenval;
    }

    // Project to faces
    template <int Direction, typename Tensor>
    static auto project_to_faces(
        const Tensor&        dofs,
        const Tensor&        flux,
        [[maybe_unused]] int sign
    )
    {
        // std::cout << "  project_to_faces: Direction=" << Direction << " sign=" << sign
        //           << " kernel=" << kernels[sign] << "\n";
        auto dofs_face =
            amr::containers::algorithms::tensor::template contract<Direction>(
                dofs, kernels[sign]
            );
        auto flux_face =
            amr::containers::algorithms::tensor::template contract<Direction>(
                flux, kernels[sign]
            );
        // std::cout << "  dofs_face result: " << dofs_face << "\n";
        return std::make_tuple(dofs_face, flux_face);
    }

    // Evaluate face integral
    template <typename Tensor>
    static auto evaluate_face_integral(
        double        maxeigenval,
        const Tensor& dofs_face,
        const Tensor& dofs_face_neigh,
        const Tensor& flux_face,
        const Tensor& flux_face_neigh,
        std::size_t   direction,
        int           sign,
        int           sign_idx,
        double        surface
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
            maxeigenval,
            numericalflux
        );
        auto kernel_vec = kernels[sign_idx];
        auto weighted_flux =
            amr::containers::algorithms::tensor::tensor_dot(numericalflux, surface_mass);
        // std::cout << "surfacemass" << surface_mass << "\n kernelvec" << kernel_vec
        //           << "\n";
        return amr::containers::algorithms::tensor::tensor_product(
            weighted_flux, kernel_vec
        );
    }
};

} // namespace amr::surface
