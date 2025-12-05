#pragma once

#include "../../build/generated_config.hpp"
#include "../containers/container_manipulations.hpp"
#include "../containers/container_operations.hpp"
#include "../containers/static_tensor.hpp"
#include "../containers/static_vector.hpp"
#include "equations/equation_impl.hpp"
#include "globals.hpp"
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
    numericalflux      = sign *
                    ((flux_face + flux_face_neigh) * 0.5 +
                     (dofs_face - dofs_face_neigh) * (0.5 * maxeigenval)) *
                    surface;
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
    auto dofs_face =
        amr::containers::manipulators::contract<Direction>(dofs, kernels[sign]);
    auto flux_face =
        amr::containers::manipulators::contract<Direction>(flux, kernels[sign]);
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
    double&                      maxeigenval,
    const EquationType&          eq,
    const auto&                  kernels,
    const Tensor&                dofs_face,
    const Tensor&                dofs_face_neigh,
    const Tensor&                flux_face,
    const Tensor&                flux_face_neigh,
    [[maybe_unused]] std::size_t direction,
    int                          sign,
    double                       surface
)
{
    using value_t       = typename Tensor::value_type;
    using face_result_t = amr::containers::utils::types::tensor::
        hypercube_t<value_t, amr::config::Order, amr::config::Dim>;

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

    auto kernel_vec = kernels[sign];

    // Apply quadrature weights along non-face dimensions
    // For 2D: apply weights along dimension 0 (non-face)
    // For 3D: apply weights along dimensions 0,1 (non-faces)
    if constexpr (amr::config::Dim == 2)
    {
        // In 2D, we have one non-face dimension (d=0)
        const auto& weights_array = amr::global::QuadData<amr::config::Order>::weights;
        amr::containers::static_vector<double, amr::config::Order> weights_vec;
        for (unsigned int i = 0; i < static_cast<unsigned int>(amr::config::Order); ++i)
        {
            weights_vec[i] = weights_array[i];
        }
        numericalflux =
            amr::containers::manipulators::einsum_apply<0>(numericalflux, weights_vec);
    }
    else if constexpr (amr::config::Dim == 3)
    {
        // In 3D, we have two non-face dimensions (d=0, d=1)
        const auto& weights_array = amr::global::QuadData<amr::config::Order>::weights;
        amr::containers::static_vector<double, amr::config::Order> weights_vec;
        for (unsigned int i = 0; i < static_cast<unsigned int>(amr::config::Order); ++i)
        {
            weights_vec[i] = weights_array[i];
        }
        numericalflux =
            amr::containers::manipulators::einsum_apply<0>(numericalflux, weights_vec);
        numericalflux =
            amr::containers::manipulators::einsum_apply<1>(numericalflux, weights_vec);
    }

    return outer_product<face_result_t>(kernel_vec, numericalflux);
}

} // namespace amr::surface
