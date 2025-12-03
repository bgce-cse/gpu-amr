#pragma once

#include "../../build/generated_config.hpp"
#include "../containers/container_manipulations.hpp"
#include "../containers/container_operations.hpp"
#include "../containers/static_tensor.hpp"
#include "../containers/static_vector.hpp"
#include "equations/equation_impl.hpp"
#include "globals.hpp"
#include <cassert>

namespace amr::surface
{
/**
 * @brief Generic buffer structure for face integrals with configurable dimension.
 *
 * Stores all temporary buffers needed during face integrals.
 * Avoids costly re-allocation by pre-allocating all required tensors
 * for face integral computations. Uses tensor-based storage for DOF values.
 *
 * @tparam Dim Spatial dimension (2 or 3)
 */
using dof_value_t = amr::containers::static_vector<double, amr::config::DOFs>;
using face_t      = typename amr::containers::utils::types::tensor::
    hypercube_t<dof_value_t, amr::config::Order, amr::config::Dim - 1>;

using dof_t = typename amr::containers::utils::types::tensor::
    hypercube_t<dof_value_t, amr::config::Order, amr::config::Dim>;

/**
 * @brief Computes the Rusanov (local Lax-Friedrichs) numerical flux.
 */

template <typename EquationType>
double rusanov(
    const EquationType& eq,
    const face_t&       dofs_face,
    const face_t&       dofs_face_neigh,
    const face_t&       flux_face,
    const face_t&       flux_face_neigh,
    double              surface,
    int                 direction,
    int                 sign,
    face_t&             numericalflux
)
{
    double eigenvalue{};

    double maxeigenval = std::max(
        eq.max_eigenvalue(dofs_face, direction),
        eq.max_eigenvalue(dofs_face_neigh, direction)
    );

    numericalflux = ((flux_face + flux_face_neigh) * 0.5 +
                     (dofs_face - dofs_face_neigh) * (0.5 * maxeigenval)) *
                    surface;

    eigenvalue = maxeigenval * surface;
    return eigenvalue;
}

/**
 * @brief Projects DOFs and flux to a face using contraction.
 */
template <typename KernelsType, int direction>
std::tuple<face_t, face_t> project_to_faces(
    const KernelsType& kernels,
    const dof_t&       dofs,
    const dof_t&       flux,
    auto               sign
)
{
    // Contract DOFs along the specified direction using kernels
    face_t dofsface =
        amr::containers::manipulators::contract<direction>(dofs, kernels[sign]);

    // Contract flux along the specified direction using kernels
    face_t fluxface =
        amr::containers::manipulators::contract<direction>(flux, kernels[sign]);

    return std::make_tuple(dofsface, fluxface);
}

/**
 * @brief Evaluates the face integral contribution to cell residuals.
 */
template <typename EquationType>
double evaluate_face_integral(
    const EquationType& eq,
    const auto&         kernels,
    const face_t&       dofs_face,
    const face_t&       dofs_face_neigh,
    const face_t&       flux_face,
    const face_t&       flux_face_neigh,
    int                 direction,
    int                 sign,
    int                 surface

)
{
    face_t numericalflux{};
    double maxeigenval = rusanov(
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

    // celldu .-= globals.project_dofs_from_face[face] * (buffers.numericalflux' *
    // globals.reference_massmatrix_face)'

    return maxeigenval;
}
} // namespace amr::surface