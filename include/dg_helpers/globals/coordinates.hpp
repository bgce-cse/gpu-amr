#ifndef AMR_GLOBAL_COORDINATES_HPP
#define AMR_GLOBAL_COORDINATES_HPP

#include "containers/container_algorithms.hpp"
#include "iostream"
#include "morton/morton_id.hpp"

namespace amr::global
{

/*======================================================================
  Type aliases for readability
======================================================================*/

template <std::size_t Dim>
using idx_vector = amr::containers::static_vector<std::size_t, Dim>;

template <std::size_t Dim>
using coord_vector = amr::containers::static_vector<double, Dim>;

/*======================================================================
  Compile-time arithmetic helpers
======================================================================*/

template <typename Vec>
constexpr double product(const Vec& v)
{
    double res = 1.0;
    for (auto x : v)
        res *= x;
    return res;
}

/*======================================================================
  Coordinate transformations
======================================================================*/

template <typename CenterType, typename RefType, typename SizeType>
constexpr auto reference_to_global(
    const CenterType& cell_center,
    const RefType&    ref_coords,
    const SizeType&   cell_size
)
{
    return cell_center + ref_coords * cell_size;
}

template <typename CenterType, typename GlobalType, typename SizeType>
constexpr auto global_to_reference(
    const CenterType& cell_center,
    const GlobalType& global_coords,
    const SizeType&   cell_size
)
{
    return (global_coords - cell_center) / cell_size;
}

template <typename PatchIndexType, std::size_t PatchSize>
constexpr auto edge(const PatchIndexType& patch_id)
{
    auto [patch_coords, patch_level] = PatchIndexType::decode(patch_id.id());
    double patch_size                = 1.0 / static_cast<double>(1u << patch_level);
    return patch_size / PatchSize;
}

template <typename SizeType>
constexpr auto volume(const SizeType& cell_size)
{
    double res = 1.0;
    for (std::size_t i = 0; i < amr::config::GlobalConfigPolicy::Dim; ++i)
        res *= cell_size;
    return res;
}

template <typename SizeType>
constexpr auto area(const SizeType& cell_size)
{
    double res = 1.0;
    for (std::size_t i = 1; i < amr::config::GlobalConfigPolicy::Dim; ++i)
        res *= cell_size;
    return res;
}

/*======================================================================
  Linear index <-> Local coordinates
======================================================================*/

template <std::size_t Dim, std::size_t PatchSize, std::size_t HaloWidth>
constexpr auto linear_to_local_coords(std::size_t linear_idx)
{
    constexpr std::size_t stride = PatchSize + 2 * HaloWidth;
    idx_vector<Dim>       coords{};

    if constexpr (Dim == 2)
    {
        coords[0] = linear_idx % stride;
        coords[1] = linear_idx / stride;
    }
    else if constexpr (Dim == 3)
    {
        const std::size_t slice = stride * stride;
        coords[0]               = linear_idx % stride;
        coords[1]               = (linear_idx / stride) % stride;
        coords[2]               = linear_idx / slice;
    }
    else
    {
        std::size_t rem = linear_idx;
        for (std::size_t d = 0; d < Dim; ++d)
        {
            coords[d] = rem % stride;
            rem /= stride;
        }
    }
    return coords;
}

template <std::size_t Dim, std::size_t HaloWidth>
constexpr auto remove_halo(const idx_vector<Dim>& coords_with_halo)
{
    idx_vector<Dim> coords{};
    for (std::size_t d = 0; d < Dim; ++d)
        coords[d] = coords_with_halo[d] - HaloWidth;
    return coords;
}

/*======================================================================
  Compute cell center within patch
======================================================================*/

template <
    std::size_t PatchSize,
    std::size_t HaloWidth,
    std::size_t Dim,
    typename PatchIndexType>
constexpr auto compute_cell_center(
    const PatchIndexType&  patch_id,
    const idx_vector<Dim>& local_indices
)
{
    const auto [patch_coords, patch_level] = PatchIndexType::decode(patch_id.id());
    constexpr double level_scale           = 1.0;
    double patch_size_factor = level_scale / static_cast<double>(1u << patch_level);
    double cell_size         = patch_size_factor / static_cast<double>(PatchSize);

    coord_vector<Dim> cell_center{};
    for (std::size_t d = 0; d < Dim; ++d)
        cell_center[d] = (static_cast<double>(patch_coords[d]) * PatchSize +
                          static_cast<double>(local_indices[d]) + 0.5) *
                         cell_size;

    return cell_center;
}

} // namespace amr::global

#endif // AMR_GLOBAL_COORDINATES_HPP
