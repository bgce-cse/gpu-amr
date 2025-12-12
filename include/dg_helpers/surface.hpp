#pragma once

#include "containers/container_algorithms.hpp"
#include "containers/static_vector.hpp"
#include "generated_config.hpp"
#include "globals/globals.hpp"
#include <cassert>
#include <tuple>

namespace amr::surface
{

// -----------------------------
// Concepts
// -----------------------------
template <typename T>
concept StaticTensor = requires(T t) {
    typename T::value_type;
    typename T::layout_t;
    { T::rank() } -> std::convertible_to<std::size_t>;
};

// -----------------------------
// Tensor traits
// -----------------------------
template <StaticTensor Tensor>
struct tensor_traits
{
    using value_type = typename Tensor::value_type;
    using layout_t   = typename Tensor::layout_t;
    using shape_t    = typename layout_t::shape_t;

    static constexpr std::size_t rank  = layout_t::rank();
    static constexpr std::size_t order = shape_t::first();
};

// -----------------------------
// Rusanov flux
// -----------------------------
template <typename EquationType, StaticTensor Tensor>
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
    numericalflux      = (sign * (flux_face + flux_face_neigh) * 0.5 +
                     (dofs_face - dofs_face_neigh) * (0.5 * maxeigenval)) *
                    surface;
    std::cout << "sign: " << sign << " numerical flux: " << numericalflux << "\n"
              << "dofs_face: " << dofs_face << " flux_face: " << flux_face << "\n"
              << "dofs_neigh: " << dofs_face_neigh
              << " flux_face_neigh: " << flux_face_neigh << "\n"
              << "surface: " << surface << " maxeigenval: " << maxeigenval << "\n";
    return maxeigenval;
}

// -----------------------------
// Project to faces
// -----------------------------
template <typename KernelsType, int Direction, StaticTensor Tensor>
auto project_to_faces(
    const KernelsType&   kernels,
    const Tensor&        dofs,
    const Tensor&        flux,
    [[maybe_unused]] int sign
)
{
    auto dofs_face = amr::containers::algorithms::tensor::template contract<Direction>(
        dofs, kernels[sign]
    );
    auto flux_face = amr::containers::algorithms::tensor::template contract<Direction>(
        flux, kernels[sign]
    );
    return std::make_tuple(dofs_face, flux_face);
}

// -----------------------------
// Outer product helper
// -----------------------------
template <StaticTensor ResultTensor, typename Vec1, typename Vec2>
ResultTensor outer_product(const Vec1& a, const Vec2& b)
{
    using index_t = typename ResultTensor::multi_index_t;

    ResultTensor result = ResultTensor::zero();
    index_t      idx{};
    do
    {
        result[idx] = a[idx[0]] * b[idx[1]];
    } while (idx.increment());
    return result;
}

// -----------------------------
// Evaluate face integral
// -----------------------------
template <typename EquationType, StaticTensor Tensor>
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
    int                 sign_idx,
    double              surface,
    const auto&         globals
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

    auto kernel_vec = kernels[sign_idx];

    // Apply quadrature weights along non-face dimensions
    // For 2D: apply weights along dimension 0 (non-face)
    // For 3D: apply weights along dimensions 0,1 (non-faces)
    if constexpr (amr::config::Dim == 2)
    {
        // In 2D, we have one non-face dimension (d=0)
        const auto& weights_array = globals.basis.quadweights();
        amr::containers::static_vector<double, amr::config::Order> weights_vec;
        for (auto i = 0u; i < amr::config::Order; ++i)
        {
            weights_vec[i] = weights_array[i];
        }
        numericalflux = amr::containers::algorithms::tensor::template einsum_apply<0>(
            numericalflux, weights_vec
        );
    }
    else if constexpr (amr::config::Dim == 3)
    {
        // In 3D, we have two non-face dimensions (d=0, d=1)
        const auto& weights_array = globals.basis.quadweights();
        amr::containers::static_vector<double, amr::config::Order> weights_vec;
        for (auto i = 0u; i < amr::config::Order; ++i)
        {
            weights_vec[i] = weights_array[i];
        }
        numericalflux = amr::containers::algorithms::tensor::template einsum_apply<0>(
            numericalflux, weights_vec
        );
        numericalflux = amr::containers::algorithms::tensor::template einsum_apply<1>(
            numericalflux, weights_vec
        );
    }
    // Combine kernel weighting with the weighted numerical flux
    // kernel_vec is 1D, tensor_dot result is already weighted
    auto weighted_flux = amr::containers::algorithms::tensor::tensor_dot(
        numericalflux, globals.surface_mass_tensors.mass_tensor
    );
    // Apply kernel weighting to each component
    return amr::containers::algorithms::tensor::template einsum_apply<0>(
        weighted_flux, kernel_vec
    );
}

} // namespace amr::surface
