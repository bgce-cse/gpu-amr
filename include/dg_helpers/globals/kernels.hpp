#ifndef AMR_GLOBAL_KERNELS_HPP
#define AMR_GLOBAL_KERNELS_HPP

#include "../basis/polynomial.hpp"
#include "containers/container_algorithms.hpp"
#include "quadrature.hpp"
#include <array>

namespace amr::global
{

/**
 * @brief Precomputed face kernels at coordinates 0 and 1 (fully compile-time)
 */
template <unsigned int Order, unsigned int Dim>
struct FaceKernels
{
    static constexpr auto kernels = []()
    {
        std::array<amr::containers::static_vector<double, Order>, 2> k{};
        k[0] = amr::basis::Basis<Order, Dim>::create_face_kernel(0.0);
        k[1] = amr::basis::Basis<Order, Dim>::create_face_kernel(1.0);
        return k;
    }();

    [[nodiscard]]
    static constexpr const auto& operator[](int idx)
    {
        return kernels[idx];
    }
};

/**
 * @brief Mass tensors (volume and surface) fully compile-time
 */
template <unsigned int Order, unsigned int Dim>
struct MassTensors
{
    static constexpr auto mass_tensor = []()
    {
        return amr::containers::algorithms::tensor::tensor_power<Dim>(
            amr::basis::Basis<Order, Dim>::quadweights
        );
    }();

    static constexpr auto inv_mass_tensor = []()
    {
        auto w = amr::basis::Basis<Order, Dim>::quadweights;
        for (auto& x : w)
            x = 1.0 / x;
        return amr::containers::algorithms::tensor::tensor_power<Dim>(w);
    }();
};

} // namespace amr::global

#endif // AMR_GLOBAL_KERNELS_HPP
