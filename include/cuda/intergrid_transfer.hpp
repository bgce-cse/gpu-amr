#ifndef AMR_INCLUDED_CUDA_INTERGRID_TRANSFER
#define AMR_INCLUDED_CUDA_INTERGRID_TRANSFER

#include <array>
#include <cstddef>
#include <cstdint>

namespace amr::cuda
{

struct intergrid_transfer_task_metadata
{
    std::int32_t source_patch      = -1;
    std::int32_t destination_patch = -1;
};

struct intergrid_transfer_launch_config
{
    std::size_t patch_flat_size;
    std::size_t data_flat_size;
    std::size_t halo_width;
    std::array<std::size_t, 3> padded_strides = { 1, 1, 1 };
    std::array<std::size_t, 3> data_sizes     = { 1, 1, 1 };
    std::array<std::size_t, 3> data_strides   = { 1, 1, 1 };
};

template <std::size_t Rank>
auto interpolate_scalar_patches_inplace(
    double*                                 device_patch_data,
    const intergrid_transfer_task_metadata* device_tasks,
    std::size_t                             task_count,
    const intergrid_transfer_launch_config& config
) -> void;

template <std::size_t Rank>
auto restrict_scalar_patches_inplace(
    double*                                 device_patch_data,
    const intergrid_transfer_task_metadata* device_tasks,
    std::size_t                             task_count,
    const intergrid_transfer_launch_config& config
) -> void;

} // namespace amr::cuda

#endif // AMR_INCLUDED_CUDA_INTERGRID_TRANSFER
