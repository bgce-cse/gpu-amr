#ifndef AMR_GLOBAL_CONFIG_HPP
#define AMR_GLOBAL_CONFIG_HPP

#include "coordinates.hpp"
#include "dg_helpers/basis/basis.hpp"
#include "dg_helpers/equations/equations.hpp"
#include "dg_patches.hpp"
#include "generated_config.hpp"
#include "globals.hpp"
#include "kernels.hpp"
#include "quadrature.hpp"

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
    std::integral auto        Order,
    std::integral auto        Dim,
    std::integral auto        NumDOF,
    std::integral auto        PatchSize,
    std::integral auto        HaloWidth,
    amr::config::EquationType EqTypeValue>
struct GlobalConfig
{
    // -------------------------------------------------------------------
    // Equation type
    // -------------------------------------------------------------------
    static constexpr amr::config::EquationType equation_type = EqTypeValue;

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
    using EqType =
        typename amr::equations::EquationTraits<EqTypeValue, NumDOF, Order, Dim>;
    using EquationImpl = typename EqType::type;

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

    static constexpr auto lin_to_local(auto linear_idx)
    {
        return linear_to_local_coords<Dim, PatchSize, HaloWidth>(linear_idx);
    }

    static constexpr auto
        rm_halo(const amr::containers::static_vector<unsigned int, Dim>& coords_with_halo)
    {
        return remove_halo<Dim, HaloWidth>(coords_with_halo);
    }

    template <typename PatchIndexType>
    static constexpr auto compute_center(
        const PatchIndexType&                                    patch_id,
        const amr::containers::static_vector<unsigned int, Dim>& local_idx
    )
    {
        return compute_cell_center<PatchSize, HaloWidth, Dim>(patch_id, local_idx);
    }

    static auto interpolate_initial_dofs(const auto& cell_center, double cell_size)
    {
        return Basis::project_to_reference_basis(
            [&](auto node_coords)
            {
                auto shifted    = node_coords - 0.5;
                auto global_pos = reference_to_global(cell_center, shifted, cell_size);
                return EquationImpl::get_initial_values(global_pos, 0.0);
            }
        );
    }
};

} // namespace amr::global

#endif // AMR_GLOBAL_CONFIG_HPP
