#pragma once

#include "../../build/generated_config.hpp"
#include "../containers/container_manipulations.hpp"
#include "../containers/container_operations.hpp"
#include "../containers/static_tensor.hpp"
#include "../containers/static_vector.hpp"
#include "equations/equation_impl.hpp"
#include "globals.hpp"
#include <cassert>
#include <iostream>
#include <tuple>

namespace amr::surface
{

// Generic traits for static_tensor
template <typename Tensor>
struct tensor_traits;

template <typename T, amr::containers::concepts::StaticLayout Layout>
struct tensor_traits<amr::containers::static_tensor<T, Layout>>
{
    using value_type = T;
    using layout_t   = Layout;
    using shape_t    = typename Layout::shape_t;

    static constexpr auto rank  = Layout::rank();
    static constexpr auto order = shape_t::first(); // adjust if shape stores uniform size
};

/**
 * @brief Computes the Rusanov (local Lax-Friedrichs) numerical flux.
 * Works with any static_tensor type.
 */
template <typename EquationType, typename Tensor>
double rusanov(
    [[maybe_unused]] const EquationType& eq,
    const Tensor&                        dofs_face,
    const Tensor&                        dofs_face_neigh,
    const Tensor&                        flux_face,
    const Tensor&                        flux_face_neigh,
    double                               surface,
    [[maybe_unused]] std::size_t         direction,
    int                                  sign,
    Tensor&                              numericalflux
)
{
    double maxeigenval = 1.0; // placeholder
    numericalflux      = sign *
                    ((flux_face + flux_face_neigh) * 0.5 +
                     (dofs_face - dofs_face_neigh) * (0.5 * maxeigenval)) *
                    surface;

    return maxeigenval * surface;
}

/**
 * @brief Projects DOFs and flux to a face using contraction.
 * Returns rank-reduced tensors (contracting along specified direction).
 */
template <typename KernelsType, int direction, typename Tensor>
auto project_to_faces(
    const KernelsType& kernels,
    const Tensor&      dofs,
    const Tensor&      flux,
    auto               sign
)
{
    auto dofsface =
        amr::containers::manipulators::contract<direction>(dofs, kernels[sign]);
    auto fluxface =
        amr::containers::manipulators::contract<direction>(flux, kernels[sign]);

    return std::make_tuple(dofsface, fluxface);
}

/**
 * @brief Evaluates the face integral contribution to cell residuals.
 * Generic over any static_tensor type.
 */
template <typename EquationType, typename Tensor>
auto evaluate_face_integral(
    double&             maxeigenval,
    const EquationType& eq,
    const auto&         kernels,
    const Tensor&       dofs_face,
    const Tensor&       dofs_face_neigh,
    const Tensor&       flux_face,
    const Tensor&       flux_face_neigh,
    std::size_t         direction,
    int                 sign,
    double              surface
)
{
    Tensor numericalflux{};
    maxeigenval = rusanov(
        eq,
        dofs_face,
        dofs_face_neigh,
        flux_face,
        flux_face_neigh,
        surface,
        direction,
        sign,
        numericalflux
    );

    // Expand rank-1 flux to rank-2 by outer product with kernel
    // Result type: rank-2 tensor with same scalar type as numericalflux
    using scalar_t = typename Tensor::value_type;

    // Result is a rank-2 hypercube tensor
    using result_t = amr::containers::utils::types::tensor::
        hypercube_t<scalar_t, amr::config::Order, amr::config::Dim>;
    auto face_du = result_t::zero();

    // Compute outer product: result[i, j] = kernel[i] * numericalflux[j]
    // This gives the contribution to cell DOFs from the face integral
    auto                             kernel_vec = kernels[sign];
    typename result_t::multi_index_t idx{};
    do
    {
        face_du[idx] = kernel_vec[idx[0]] * numericalflux[idx[1]];
    } while (idx.increment());

    return face_du;
}

} // namespace amr::surface
