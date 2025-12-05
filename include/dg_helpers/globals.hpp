#ifndef GLOBAL_HPP
#define GLOBAL_HPP

#include "basis.hpp"
#include "containers/container_operations.hpp"
#include <array>

namespace amr::global
{

/**
 * @brief Pre-computed Gauss-Legendre quadrature data template on [-1, 1]
 */
template <unsigned int Order>
struct QuadData;

template <>
struct QuadData<1>
{
    static constexpr std::array<double, 1> points  = { 0.0 };
    static constexpr std::array<double, 1> weights = { 2.0 };
};

template <>
struct QuadData<2>
{
    static constexpr std::array<double, 2> points  = { -0.5773502691896257,
                                                       0.5773502691896257 };
    static constexpr std::array<double, 2> weights = { 1.0, 1.0 };
};

template <>
struct QuadData<3>
{
    static constexpr std::array<double, 3> points  = { -0.7745966692414834,
                                                       0.0,
                                                       0.7745966692414834 };
    static constexpr std::array<double, 3> weights = { 0.5555555555555556,
                                                       0.8888888888888888,
                                                       0.5555555555555556 };
};

template <>
struct QuadData<4>
{
    static constexpr std::array<double, 4> points  = { -0.8611363115940526,
                                                       -0.3399810435848563,
                                                       0.3399810435848563,
                                                       0.8611363115940526 };
    static constexpr std::array<double, 4> weights = { 0.3478548451374538,
                                                       0.6521451548625461,
                                                       0.6521451548625461,
                                                       0.3478548451374538 };
};

template <>
struct QuadData<5>
{
    static constexpr std::array<double, 5> points  = { -0.9061798459386640,
                                                       -0.5384693101056831,
                                                       0.0,
                                                       0.5384693101056831,
                                                       0.9061798459386640 };
    static constexpr std::array<double, 5> weights = { 0.2369268850561891,
                                                       0.4786286704993665,
                                                       0.5688888888888889,
                                                       0.4786286704993665,
                                                       0.2369268850561891 };
};

template <>
struct QuadData<6>
{
    static constexpr std::array<double, 6> points = {
        -0.9324695142031521, -0.6612093864662645, -0.2386191860831969,
        0.2386191860831969,  0.6612093864662645,  0.9324695142031521
    };
    static constexpr std::array<double, 6> weights = {
        0.1713244923791704, 0.3607615730481386, 0.4679139345726910,
        0.4679139345726910, 0.3607615730481386, 0.1713244923791704
    };
};

template <>
struct QuadData<7>
{
    static constexpr std::array<double, 7> points = {
        -0.9491079123427585, -0.7415311855993945, -0.4058451513773972, 0.0,
        0.4058451513773972,  0.7415311855993945,  0.9491079123427585
    };
    static constexpr std::array<double, 7> weights = {
        0.1294849661688697, 0.2797053914892766, 0.3818300505051189, 0.4179591836734694,
        0.3818300505051189, 0.2797053914892766, 0.1294849661688697
    };
};

template <>
struct QuadData<8>
{
    static constexpr std::array<double, 8> points = {
        -0.9602898564975363, -0.7966664774136267, -0.5255324099163290,
        -0.1834346424956498, 0.1834346424956498,  0.5255324099163290,
        0.7966664774136267,  0.9602898564975363
    };
    static constexpr std::array<double, 8> weights = {
        0.1012285362903763, 0.2223810344533745, 0.3137066458778873, 0.3626837833783620,
        0.3626837833783620, 0.3137066458778873, 0.2223810344533745, 0.1012285362903763
    };
};

/**
 * @brief Compile-time dictionary storing precomputed face kernels for coordinates 0
 * and 1.
 *
 * @tparam Order Polynomial order
 * @tparam Dim Spatial dimension
 */
template <unsigned int Order, unsigned int Dim>
struct FaceKernels
{
    std::array<amr::containers::static_vector<double, Order>, 2> kernels;

    /**
     * @brief Construct and precompute face kernels at compile time.
     *
     * @param basis The basis object used to compute kernels
     */
    constexpr FaceKernels(const amr::Basis::Basis<Order, Dim>& basis)
    {
        // Compute kernels for face coordinates 0.0 and 1.0
        kernels[0] = basis.create_face_kernel(0.0);
        kernels[1] = basis.create_face_kernel(1.0);
    }

    /**
     * @brief Access kernel by face coordinate index.
     */
    constexpr const amr::containers::static_vector<double, Order>&
        operator[](int idx) const
    {
        return kernels[idx];
    }
};

/**
 * @brief Convert reference coordinates to global position
 *
 * Maps reference coordinates in [0,1] to global domain coordinates.
 * Uses templating for compatibility with vector operations.
 *
 * @tparam CenterType Type of cell center
 * @tparam RefType Type of reference coordinates [0,1]
 * @tparam SizeType Type of cell size
 * @param cell_center Global coordinates of cell center
 * @param ref_coords Reference coordinates in [0,1]
 * @param cell_size Cell size in each dimension
 * @return Global coordinates
 */
template <typename CenterType, typename RefType, typename SizeType>
inline auto reference_to_global(
    const CenterType& cell_center,
    const RefType&    ref_coords,
    const SizeType&   cell_size
)
{
    return cell_center + ref_coords * cell_size;
}

/**
 * @brief Convert global coordinates to reference coordinates
 *
 * Maps global domain coordinates to reference coordinates in [0,1].
 * Uses templating for compatibility with vector operations.
 *
 * @tparam CenterType Type of cell center
 * @tparam GlobalType Type of global coordinates
 * @tparam SizeType Type of cell size
 * @param cell_center Global coordinates of cell center
 * @param global_coords Global coordinates
 * @param cell_size Cell size in each dimension
 * @return Reference coordinates in [0,1]
 */
template <typename CenterType, typename GlobalType, typename SizeType>
inline auto global_to_reference(
    const CenterType& cell_center,
    const GlobalType& global_coords,
    const SizeType&   cell_size
)
{
    return (global_coords - cell_center) / cell_size;
}

/**
 * @brief Calculate the volume of a cell
 *
 * Returns the product of all cell dimensions (cell_size).
 * For a cell in Dim dimensions, this is the Dim-dimensional volume.
 *
 * @param cell_size Cell dimensions (static_vector)
 * @return Volume as a scalar (product of all components)
 */
template <typename SizeType>
inline auto volume(const SizeType& cell_size)
{
    // Volume = product of all dimensions
    auto result = cell_size[0];
    for (unsigned int i = 1; i < cell_size.elements(); ++i)
    {
        result *= cell_size[i];
    }
    return result;
}

/**
 * @brief Calculate the area (surface measure) of a cell
 *
 * Returns the product of all cell dimensions except the first.
 * In 2D, returns the y-dimension. In 3D, returns y*z.
 * In 1D, returns 1.0.
 *
 * @param cell_size Cell dimensions (static_vector)
 * @return Area as a scalar (product of dimensions 1 to end)
 */
template <typename SizeType>
inline auto area(const SizeType& cell_size)
{
    // Area = product of all dimensions except first
    auto result = 1.0;
    for (unsigned int i = 1; i < cell_size.elements(); ++i)
    {
        result *= cell_size[i];
    }
    return result;
}

/**
 * @brief Convert linear index to multi-dimensional local coordinates
 *
 * Maps a flat linear index to multi-dimensional coordinates.
 * For 2D with stride (PatchSize + 2*HaloWidth):
 *   x = linear_idx % stride
 *   y = linear_idx / stride
 *
 * @tparam Dim Spatial dimension
 * @tparam PatchSize Number of cells per patch dimension
 * @tparam HaloWidth Width of halo region
 *
 * @param linear_idx Flat linear index
 * @return Multi-dimensional local coordinates (with halo)
 */
template <unsigned int Dim, std::size_t PatchSize, std::size_t HaloWidth>
inline amr::containers::static_vector<std::size_t, Dim>
    linear_to_local_coords(std::size_t linear_idx)
{
    constexpr std::size_t                            stride = PatchSize + 2 * HaloWidth;
    amr::containers::static_vector<std::size_t, Dim> coords;

    if constexpr (Dim == 2)
    {
        coords[0] = linear_idx % stride;
        coords[1] = linear_idx / stride;
    }
    else if constexpr (Dim == 3)
    {
        std::size_t slice_size = stride * stride;
        coords[0]              = linear_idx % stride;
        coords[1]              = (linear_idx / stride) % stride;
        coords[2]              = linear_idx / slice_size;
    }
    else
    {
        // Fallback for other dimensions
        std::size_t remaining = linear_idx;
        for (unsigned int d = 0; d < Dim; ++d)
        {
            coords[d] = remaining % stride;
            remaining /= stride;
        }
    }
    return coords;
}

/**
 * @brief Remove halo from multi-dimensional coordinates
 *
 * Converts coordinates in the halo-padded space to interior cell coordinates.
 *
 * @tparam Dim Spatial dimension
 * @tparam HaloWidth Width of halo region
 *
 * @param coords_with_halo Coordinates including halo offset
 * @return Interior cell coordinates
 */
template <unsigned int Dim, std::size_t HaloWidth>
inline amr::containers::static_vector<std::size_t, Dim>
    remove_halo(const amr::containers::static_vector<std::size_t, Dim>& coords_with_halo)
{
    amr::containers::static_vector<std::size_t, Dim> coords;
    for (unsigned int d = 0; d < Dim; ++d)
    {
        coords[d] = coords_with_halo[d] - HaloWidth;
    }
    return coords;
}

/**
 * @brief Compute cell center coordinates in global domain
 *
 * Calculates the global coordinates of a cell center given patch coordinates,
 * local cell indices, and refinement level. Supports arbitrary dimensions.
 *
 * @tparam PatchSize Number of cells per patch dimension
 * @tparam HaloWidth Width of halo region around patch
 * @tparam Dim Spatial dimension (2 or 3)
 * @tparam PatchIndexType Type of patch index (morton_id, etc.)
 *
 * @param patch_id Patch index containing coordinates and level
 * @param local_indices Local cell coordinates within patch (0 to PatchSize-1)
 * @return Cell center coordinates in global domain
 */
template <
    std::size_t  PatchSize,
    std::size_t  HaloWidth,
    unsigned int Dim,
    typename PatchIndexType>
inline amr::containers::static_vector<double, Dim> compute_cell_center(
    const PatchIndexType&                                   patch_id,
    const amr::containers::static_vector<std::size_t, Dim>& local_indices
)
{
    auto [patch_coords, patch_level] = PatchIndexType::decode(patch_id.id());
    double patch_level_size          = 1.0 / static_cast<double>(1u << patch_level);
    double cell_size                 = patch_level_size / static_cast<double>(PatchSize);

    amr::containers::static_vector<double, Dim> cell_center;
    for (unsigned int d = 0; d < Dim; ++d)
    {
        cell_center[d] =
            (static_cast<double>(patch_coords[d]) * static_cast<double>(PatchSize) +
             static_cast<double>(local_indices[d]) + 0.5) *
            cell_size;
    }
    return cell_center;
}

} // namespace amr::global

#endif // GLOBAL_HPP