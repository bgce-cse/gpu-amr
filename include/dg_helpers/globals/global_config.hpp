#ifndef AMR_GLOBAL_CONFIG_HPP
#define AMR_GLOBAL_CONFIG_HPP

#include "coordinates.hpp"
#include "dg_helpers/basis/basis.hpp"
#include "dg_helpers/equations/equations.hpp"
#include "dg_helpers/time_integration/time_integration.hpp"
#include "generated_config.hpp"
#include "globals.hpp"
#include "kernels.hpp"
#include "quadrature.hpp"
#include <type_traits>

namespace amr::global
{

template <typename Policy>
struct QuadratureMixin
{
    static constexpr auto& quad_points  = QuadData<Policy::Order>::points;
    static constexpr auto& quad_weights = QuadData<Policy::Order>::weights;
};

template <typename Policy>
struct BasisMixin
{
    using Basis    = amr::basis::Basis<Policy::Order, Policy::Dim>;
    using Lagrange = typename Basis::Lagrange_t;
};

template <typename Policy>
struct EquationMixin
{
    static constexpr auto equation_type = Policy::equation;

    using EqTraits = typename amr::equations::
        EquationTraits<equation_type, Policy::DOFs, Policy::Order, Policy::Dim>;

    using EquationImpl = typename EqTraits::type;
};

template <typename Policy>
struct IntegratorMixin
{
    // Note: IntegratorType depends on the actual patch container type,
    // which is only known at runtime in main(). Store the enum for later use.
    static constexpr auto integrator_type = Policy::integrator;
};

template <typename Policy>
struct TensorMixin
{
    using MassTensors        = amr::global::MassTensors<Policy::Order, Policy::Dim>;
    using SurfaceMassTensors = amr::global::MassTensors<Policy::Order, Policy::Dim - 1>;
    using FaceKernels        = amr::global::FaceKernels<Policy::Order, Policy::Dim>;

    static constexpr auto& volume_mass      = MassTensors::mass_tensor;
    static constexpr auto& inv_volume_mass  = MassTensors::inv_mass_tensor;
    static constexpr auto& surface_mass     = SurfaceMassTensors::mass_tensor;
    static constexpr auto& inv_surface_mass = SurfaceMassTensors::inv_mass_tensor;
    static constexpr auto& face_kernels     = FaceKernels::kernels;
};

template <typename Policy>
struct CoordinateMixin
{
    template <typename Center, typename Ref, typename Size>
    static constexpr auto ref_to_global(const Center& c, const Ref& r, const Size& s)
    {
        return reference_to_global(c, r, s);
    }

    template <typename Center, typename Global, typename Size>
    static constexpr auto global_to_ref(const Center& c, const Global& g, const Size& s)
    {
        return global_to_reference(c, g, s);
    }

    static auto cell_edge(std::integral auto& idx)
    {
        return edge(idx);
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

    static constexpr auto lin_to_local(const auto& idx)
    {
        return linear_to_local_coords<Policy::Dim, Policy::PatchSize, Policy::HaloWidth>(
            idx
        );
    }

    static constexpr auto rm_halo(const auto& coords)
    {
        return remove_halo<Policy::Dim, Policy::HaloWidth>(coords);
    }

    template <typename PatchIndexType>
    static constexpr auto
        compute_center(const PatchIndexType& patch_id, const auto& local)
    {
        return compute_cell_center<Policy::PatchSize, Policy::HaloWidth, Policy::Dim>(
            patch_id, local
        );
    }
};

template <typename Policy>
struct InitMixin
{
private:
    using Basis  = amr::basis::Basis<Policy::Order, Policy::Dim>;
    using EqImpl = typename amr::equations::
        EquationTraits<Policy::equation, Policy::DOFs, Policy::Order, Policy::Dim>::type;

public:
    static constexpr auto interpolate_initial_dofs(const auto& center, const auto& size)
    {
        auto coeffs = Basis::project_to_reference_basis(
            [&](auto node_coords)
            {
                auto shifted    = node_coords - 0.5;
                auto global_pos = reference_to_global(center, shifted, size);
                return EqImpl::get_initial_values(global_pos, 0.0);
            }
        );
        // Apply mass matrix inverse (L2 projection formula)
        // coeffs = M^{-1} * coeffs, where M is the mass matrix
        return amr::containers::algorithms::tensor::tensor_dot(
            coeffs, TensorMixin<Policy>::inv_volume_mass
        );
    }
};

template <typename Policy>
struct GlobalConfig :
    QuadratureMixin<Policy>,
    BasisMixin<Policy>,
    EquationMixin<Policy>,
    IntegratorMixin<Policy>,
    TensorMixin<Policy>,
    CoordinateMixin<Policy>,
    InitMixin<Policy>
{
    // Bring in types from mixins for easier access
    using Basis              = typename BasisMixin<Policy>::Basis;
    using Lagrange           = typename BasisMixin<Policy>::Lagrange;
    using EquationImpl       = typename EquationMixin<Policy>::EquationImpl;
    using MassTensors        = typename TensorMixin<Policy>::MassTensors;
    using SurfaceMassTensors = typename TensorMixin<Policy>::SurfaceMassTensors;
    using FaceKernels        = typename TensorMixin<Policy>::FaceKernels;

    // Bring in static data members
    using QuadratureMixin<Policy>::quad_points;
    using QuadratureMixin<Policy>::quad_weights;
    using TensorMixin<Policy>::volume_mass;
    using TensorMixin<Policy>::inv_volume_mass;
    using TensorMixin<Policy>::surface_mass;
    using TensorMixin<Policy>::inv_surface_mass;
    using TensorMixin<Policy>::face_kernels;
};

} // namespace amr::global

#endif // AMR_GLOBAL_CONFIG_HPP
