#ifndef AMR_GLOBAL_CONFIG_HPP
#define AMR_GLOBAL_CONFIG_HPP

#include "coordinates.hpp"
#include "dg_helpers/basis/basis.hpp"
#include "dg_helpers/equations/advection.hpp"
#include "dg_helpers/equations/equation_impl.hpp"
#include "dg_helpers/equations/euler.hpp"
#include "globals.hpp"
#include "kernels.hpp"

namespace amr::global
{

/**
 * @brief Compile-time global configuration for DG methods
 *
 * Provides access to basis, quadrature, kernels, and coordinate utilities.
 *
 * @tparam Order     Polynomial order for DG basis
 * @tparam Dim       Spatial dimension
 * @tparam PatchSize Number of cells per patch dimension
 * @tparam HaloWidth Width of halo region
 */
template <
    std::size_t  Order,
    unsigned int Dim,
    std::size_t  PatchSize,
    std::size_t  HaloWidth>
struct GlobalConfig
{
    // -------------------------------------------------------------------
    // Compile-time quadrature data
    // -------------------------------------------------------------------
    static constexpr const auto& quad_points  = QuadData<Order>::points;
    static constexpr const auto& quad_weights = QuadData<Order>::weights;

    // -------------------------------------------------------------------
    // Basis and kernels
    // -------------------------------------------------------------------
    using Basis              = amr::basis::Basis<Order, Dim>;
    using Lagrange           = typename Basis::Lagrange_t;
    using FaceKernels        = amr::global::FaceKernels<Order, Dim>;
    using MassTensors        = amr::global::MassTensors<Order, Dim>;
    using SurfaceMassTensors = amr::global::MassTensors<Order, Dim - 1>;

    // Precomputed tensors (compile-time)
    static constexpr auto volume_mass      = MassTensors::mass_tensor;
    static constexpr auto inv_volume_mass  = MassTensors::inv_mass_tensor;
    static constexpr auto surface_mass     = SurfaceMassTensors::mass_tensor;
    static constexpr auto inv_surface_mass = SurfaceMassTensors::inv_mass_tensor;
    static constexpr auto face_kernels     = FaceKernels::kernels;

    // -------------------------------------------------------------------
    // Coordinate helpers (fully constexpr)
    // -------------------------------------------------------------------
    template <typename CenterType, typename RefType, typename SizeType>
    static constexpr auto
        ref_to_global(const CenterType& center, const RefType& ref, const SizeType& size)
    {
        return reference_to_global(center, ref, size);
    }

    template <typename CenterType, typename GlobalType, typename SizeType>
    static constexpr auto global_to_ref(
        const CenterType& center,
        const GlobalType& global,
        const SizeType&   size
    )
    {
        return global_to_reference(center, global, size);
    }

    template <typename SizeType>
    static constexpr auto cell_volume(const SizeType& size)
    {
        return volume(size);
    }

    template <typename SizeType>
    static constexpr auto cell_area(const SizeType& size)
    {
        return area(size);
    }

    static constexpr auto lin_to_local(std::size_t linear_idx)
    {
        return linear_to_local_coords<Dim, PatchSize, HaloWidth>(linear_idx);
    }

    static constexpr auto
        rm_halo(const amr::containers::static_vector<std::size_t, Dim>& coords_with_halo)
    {
        return remove_halo<Dim, HaloWidth>(coords_with_halo);
    }

    template <typename PatchIndexType>
    static constexpr auto compute_center(
        const PatchIndexType&                                   patch_id,
        const amr::containers::static_vector<std::size_t, Dim>& local_idx
    )
    {
        return compute_cell_center<PatchSize, HaloWidth, Dim>(patch_id, local_idx);
    }
};

} // namespace amr::global

#endif // AMR_GLOBAL_CONFIG_HPP
