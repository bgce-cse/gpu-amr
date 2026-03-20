#ifndef AMR_INCLUDED_CUDA_HALO_EXCHANGE
#define AMR_INCLUDED_CUDA_HALO_EXCHANGE

#include <array>
#include <cstddef>
#include <cstdint>

namespace amr::cuda
{

enum class halo_neighbor_relation : std::int8_t
{
    none    = 0,
    same    = 1,
    finer   = 2,
    coarser = 3,
};

struct halo_direction_metadata
{
    std::int32_t           neighbor        = -1;
    std::int32_t           finer_ids[4]    = { -1, -1, -1, -1 };
    std::int32_t           quadrant[3]     = { 0, 0, 0 };
    halo_neighbor_relation relation        = halo_neighbor_relation::none;
    std::int8_t            padding[3]      = { 0, 0, 0 };
};

struct halo_exchange_launch_config
{
    std::size_t num_patches;
    std::size_t patch_flat_size;
    std::size_t rank;
    std::size_t halo_width;
    std::size_t fanout_1d;
    std::array<std::size_t, 3> padded_sizes   = { 1, 1, 1 };
    std::array<std::size_t, 3> padded_strides = { 1, 1, 1 };
    std::array<std::size_t, 3> data_sizes     = { 1, 1, 1 };
    std::array<std::size_t, 3> halo_cells_per_dim = { 1, 1, 1 };
    std::size_t halo_work_items_per_patch = 0;
};

auto halo_exchange_scalar_patches_inplace(
    double*                            device_patch_data,
    const halo_direction_metadata*     device_neighbor_metadata,
    std::size_t                        metadata_count,
    const halo_exchange_launch_config& config
) -> void;

} // namespace amr::cuda

#endif // AMR_INCLUDED_CUDA_HALO_EXCHANGE
