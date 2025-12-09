#ifndef DG_HELPERS_RHS_HPP
#define DG_HELPERS_RHS_HPP

#include "basis.hpp"
#include "containers/container_manipulations.hpp"
#include "containers/container_operations.hpp"
#include "equations.hpp"
#include "globals.hpp"
#include "ndtree/ndtree.hpp"
#include "ndtree/neighbor.hpp"
#include "surface.hpp"
#include <functional>
#include <utility>
#include <variant>

namespace amr::rhs
{

// Helper function to dispatch to the correct compile-time direction
template <
    std::size_t Dim,
    typename KernelsType,
    typename DOFType,
    typename FluxType,
    typename SignType,
    std::size_t... Is>
auto dispatch_project_impl(
    std::index_sequence<Is...>,
    const KernelsType& kernels,
    const DOFType&     dof,
    const FluxType&    flux,
    SignType           sign,
    std::size_t        runtime_dim
)
{
    // Use a compile-time unrolled if-else chain
    std::decay_t<decltype(amr::surface::project_to_faces<KernelsType, 0>(
        kernels, dof, flux, sign
    ))>
        result;

    bool found = false;
    ((Is == runtime_dim ? (result = amr::surface::project_to_faces<KernelsType, Is>(
                               kernels, dof, flux, sign
                           ),
                           found = true)
                        : false) ||
     ...);

    return result;
}

template <
    std::size_t Dim,
    typename KernelsType,
    typename DOFType,
    typename FluxType,
    typename SignType>
auto dispatch_project_to_faces(
    const KernelsType& kernels,
    const DOFType&     dof,
    const FluxType&    flux,
    SignType           sign,
    std::size_t        runtime_dim
)
{
    return dispatch_project_impl<Dim>(
        std::make_index_sequence<Dim>{}, kernels, dof, flux, sign, runtime_dim
    );
}

/**
 * @brief RHS Evaluator for DG discretization
 *
 * Statically caches mass tensors and compile-time constants to avoid
 * repeated allocation and computation across multiple RHS evaluations.
 */
template <
    std::size_t Order,
    std::size_t Dim,
    std::size_t PatchSize,
    std::size_t HaloWidth,
    std::size_t DOFs>
struct RHSEvaluator
{
    // Static compile-time constants
    static constexpr std::size_t order      = Order;
    static constexpr std::size_t dim        = Dim;
    static constexpr std::size_t patch_size = PatchSize;
    static constexpr std::size_t halo_width = HaloWidth;
    static constexpr std::size_t dofs       = DOFs;

    /**
     * @brief Evaluate the RHS of the DG discretization
     *
     * Computes du/dt = -∇·F(u) for the entire patch.
     *
     * @param eq Equation instance (provides flux computation)
     * @param basis DG basis (provides quadrature points and weights)
     * @param dof_patch Current DOF values for entire patch (with halo)
     * @param flux_patch Storage for computed fluxes (with halo)
     * @param patch_update Output: RHS values for entire patch (interior only)
     * @param cell_center Global coordinates of cell center
     * @param kernels Pre-computed face kernels for efficiency
     * @param dt Time step
     * @param patch_layout_t Patch layout information
     */
    template <
        typename EquationType,
        typename BasisType,
        typename DOFTensorType,
        typename FluxVectorType,
        typename KernelsType,
        typename PatchLayoutType,
        typename CenterType>
    inline static void evaluate(
        const EquationType&               eq,
        [[maybe_unused]] const BasisType& basis,
        DOFTensorType&                    dof_patch,
        FluxVectorType&                   flux_patch,
        DOFTensorType&                    patch_update,
        [[maybe_unused]] CenterType&      cell_center,
        const KernelsType&                kernels,
        [[maybe_unused]] double&          dt,
        const PatchLayoutType&            patch_layout_t,
        [[maybe_unused]] const auto&      globals,
        [[maybe_unused]] const double&    volume = 0.01
    )
    {
        using direction_t     = amr::ndt::neighbors::direction<Dim>;
        using padded_layout_t = typename PatchLayoutType::padded_layout_t;
        using flux_vector_t   = std::decay_t<decltype(flux_patch[0])>;
        using size_type       = typename flux_vector_t::size_type;

        // Loop over each cell in the patch (including halos)
        for (std::size_t linear_idx = 0; linear_idx < patch_layout_t.flat_size();
             ++linear_idx)
        {
            if (amr::ndt::utils::patches::is_halo_cell<PatchLayoutType>(linear_idx))
            {
                continue;
            }

            // Compute flux for current cell
            eq.evaluate_flux(dof_patch[linear_idx], flux_patch[linear_idx]);

            // ========== DIRECTION SYSTEM EXPLANATION ==========
            // The tree uses a NATURAL DIRECTION ORDERING based on the layout strides:
            // static_layout uses ROW-MAJOR order where the LAST dimension varies fastest
            // For 2D shape [size_0, size_1]: stride[0]=size_1, stride[1]=1
            //
            // This means direction dimensions map as follows:
            //   - Direction dimension 0 → Layout dimension 0 → NORTH/SOUTH (Y-direction)
            //   - Direction dimension 1 → Layout dimension 1 → EAST/WEST (X-direction)
            //
            // Direction iteration order and meaning (2D):
            //   d.index() = 0  →  Dimension 0, NEGATIVE direction (SOUTH, -Y)
            //   d.index() = 1  →  Dimension 0, POSITIVE direction (NORTH, +Y)
            //   d.index() = 2  →  Dimension 1, NEGATIVE direction (WEST, -X)
            //   d.index() = 3  →  Dimension 1, POSITIVE direction (EAST, +X)
            //
            // Key functions:
            //   d.dimension() = d.index() / 2        → which dimension (0 or 1)
            //   is_negative(d) = (d.index() % 2 == 0) → true if NEGATIVE (SOUTH or WEST)
            //
            // Since layout is [dim0][dim1] with dim1 fastest, NO SWAP is needed!
            // layout_dim = dim_in_direction directly
            // =====================================================

            // Loop over neighbor directions
            for (auto d = direction_t::first(); d != direction_t::sentinel(); d.advance())
            {
                // Convert current linear index to multi-index using padded_layout_t
                auto current_coords = padded_layout_t::multi_index(
                    static_cast<typename padded_layout_t::index_t>(linear_idx)
                );

                // Natural neighbor in dimension d
                auto dim_idx  = d.dimension();
                bool is_neg   = direction_t::is_negative(d);
                int  sign     = is_neg ? -1 : 1;
                int  sign_idx = is_neg ? 0 : 1;

                // Compute neighbor
                auto neighbor_coords = current_coords;
                neighbor_coords[dim_idx] += sign;

                auto neighbor_linear_idx = padded_layout_t::linear_index(neighbor_coords);

                // Bounds check using padded layout size
                if (neighbor_linear_idx >= padded_layout_t::flat_size())
                {
                    std::cout << "[BOUNDS] OUT OF BOUNDS\n";
                    continue;
                }

                auto actual_dim    = Dim - dim_idx - 1;
                auto actual_dim_sz = static_cast<size_type>(actual_dim);

                // Determine semantic direction (top/bottom for dim 0, left/right for dim
                // 1)
                // std::string_view direction =
                //     (actual_dim == 1)   ? (is_neg ? "BOTTOM" : "TOP")
                //     : (actual_dim == 0) ? (is_neg ? "LEFT" : "RIGHT")
                //                         : "UNKNOWN";

                // std::cout << "[EIGENVALUE] face=" << direction << ")\n";

                auto cell_data = dispatch_project_to_faces<Dim>(
                    kernels,
                    dof_patch[linear_idx],
                    flux_patch[linear_idx][actual_dim_sz],
                    sign_idx,
                    actual_dim
                );

                auto neighbor_data = dispatch_project_to_faces<Dim>(
                    kernels,
                    dof_patch[neighbor_linear_idx],
                    flux_patch[neighbor_linear_idx][actual_dim_sz],
                    !sign_idx,
                    actual_dim
                );

                [[maybe_unused]]
                double curr_eigenval = 1.0;
                auto   face_du       = amr::surface::evaluate_face_integral(
                    curr_eigenval,
                    eq,
                    kernels,
                    std::get<0>(cell_data),
                    std::get<0>(neighbor_data),
                    std::get<1>(cell_data),
                    std::get<1>(neighbor_data), // merda pura
                    actual_dim,
                    sign,
                    sign_idx,
                    0.1, // edge surface
                    globals
                );

                patch_update[linear_idx] = patch_update[linear_idx] - face_du;
            }
            patch_update[linear_idx] = amr::containers::algorithms::tensor::tensor_dot(
                patch_update[linear_idx],
                globals.mass_tensors.inv_mass_tensor *
                    100 // inverse of the volume, fix it or die
            );
            // std::cout << "index" << linear_idx << " patch update "
            //           << patch_update[linear_idx] << "\n";
        }
    }
};

// Legacy interface for backward compatibility
template <
    std::size_t Order,
    std::size_t Dim,
    std::size_t PatchSize,
    std::size_t HaloWidth,
    std::size_t DOFs,
    typename EquationType,
    typename BasisType,
    typename DOFTensorType,
    typename FluxVectorType,
    typename KernelsType,
    typename PatchLayoutType,
    typename CenterType>
inline void evaluate_rhs(
    const EquationType&               eq,
    [[maybe_unused]] const BasisType& basis,
    DOFTensorType&                    dof_patch,
    FluxVectorType&                   flux_patch,
    DOFTensorType&                    patch_update,
    [[maybe_unused]] CenterType&      cell_center,
    const KernelsType&                kernels,
    [[maybe_unused]] double&          dt,
    const PatchLayoutType&            patch_layout_t
)
{
    using Evaluator = RHSEvaluator<Order, Dim, PatchSize, HaloWidth, DOFs>;
    Evaluator::evaluate(
        eq,
        basis,
        dof_patch,
        flux_patch,
        patch_update,
        cell_center,
        kernels,
        dt,
        patch_layout_t
    );
}

} // namespace amr::rhs

#endif // DG_HELPERS_RHS_HPP