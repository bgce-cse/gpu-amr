#include "cuda/intergrid_transfer.hpp"

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>

namespace amr::cuda
{

namespace
{

#define AMR_DEVICE_FORCEINLINE __device__ __forceinline__

auto throw_if_cuda_error(cudaError_t status, const char* context) -> void
{
    if (status == cudaSuccess)
    {
        return;
    }

    std::ostringstream oss;
    oss << context << ": " << cudaGetErrorString(status);
    throw std::runtime_error(oss.str());
}

struct device_intergrid_transfer_launch_config
{
    std::size_t patch_flat_size;
    std::size_t data_flat_size;
    std::size_t halo_width;
    std::size_t padded_strides[3];
    std::size_t data_sizes[3];
    std::size_t data_strides[3];
};

template <std::size_t Rank>
AMR_DEVICE_FORCEINLINE auto linear_to_coords(
    std::size_t        linear_idx,
    const std::size_t* strides,
    const std::size_t* sizes,
    std::size_t*       coords
) -> void
{
    for (std::size_t d = 0; d != Rank; ++d)
    {
        coords[d] = linear_idx / strides[d];
        linear_idx %= strides[d];
        if (coords[d] >= sizes[d])
        {
            coords[d] = 0;
        }
    }
}

template <std::size_t Rank>
AMR_DEVICE_FORCEINLINE auto coords_to_linear(
    const std::size_t* coords,
    const std::size_t* strides
) -> std::size_t
{
    std::size_t linear = 0;
    for (std::size_t d = 0; d != Rank; ++d)
    {
        linear += coords[d] * strides[d];
    }
    return linear;
}

template <std::size_t Rank>
AMR_DEVICE_FORCEINLINE auto fill_transfer_mapping(
    std::size_t        interior_linear_idx,
    const device_intergrid_transfer_launch_config& config,
    std::size_t*       fine_patch_idx,
    std::size_t*       coarse_linear_idx,
    std::size_t*       fine_linear_indices
) -> void
{
    std::size_t coarse_coords[3]    = { 0, 0, 0 };
    std::size_t padded_coords[3]    = { 0, 0, 0 };
    std::size_t base_fine_coords[3] = { 0, 0, 0 };

    linear_to_coords<Rank>(
        interior_linear_idx,
        config.data_strides,
        config.data_sizes,
        coarse_coords
    );

    *fine_patch_idx = 0;
    for (std::size_t d = 0; d != Rank; ++d)
    {
        padded_coords[d] = coarse_coords[d] + config.halo_width;

        const auto section_size = config.data_sizes[d] / 2;
        const auto child_coord  = coarse_coords[d] / section_size;
        *fine_patch_idx = (*fine_patch_idx * 2) + child_coord;

        base_fine_coords[d] =
            (coarse_coords[d] % section_size) * 2 + config.halo_width;
    }

    *coarse_linear_idx =
        coords_to_linear<Rank>(padded_coords, config.padded_strides);

    constexpr auto num_fine_values = std::size_t{ 1 } << Rank;

    for (std::size_t n = 0; n != num_fine_values; ++n)
    {
        std::size_t rem            = n;
        std::size_t fine_coords[3] = {
            base_fine_coords[0], base_fine_coords[1], base_fine_coords[2]
        };

        std::size_t stride = num_fine_values;
        for (std::size_t d = 0; d != Rank; ++d)
        {
            stride /= 2;
            fine_coords[d] += rem / stride;
            rem %= stride;
        }

        fine_linear_indices[n] = coords_to_linear<Rank>(fine_coords, config.padded_strides);
    }
}

template <std::size_t Rank>
__global__ void interpolate_scalar_patches_kernel(
    double*                                    device_patch_data,
    const intergrid_transfer_task_metadata*    tasks,
    std::size_t                                task_count,
    device_intergrid_transfer_launch_config    config
)
{
    const auto global_idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto total_work = task_count * config.data_flat_size;
    if (global_idx >= total_work)
    {
        return;
    }

    const auto task_idx            = global_idx / config.data_flat_size;
    const auto interior_linear_idx = global_idx % config.data_flat_size;
    const auto task                = tasks[task_idx];

    std::size_t child_patch_offset = 0;
    std::size_t coarse_linear_idx  = 0;
    std::size_t fine_linear_indices[8] = { 0 };
    fill_transfer_mapping<Rank>(
        interior_linear_idx,
        config,
        &child_patch_offset,
        &coarse_linear_idx,
        fine_linear_indices
    );

    const auto src_patch_base =
        static_cast<std::size_t>(task.source_patch) * config.patch_flat_size;
    const auto dst_patch_base = static_cast<std::size_t>(task.destination_patch +
                                                         static_cast<std::int32_t>(child_patch_offset)) *
                                config.patch_flat_size;
    const auto value = device_patch_data[src_patch_base + coarse_linear_idx];

    constexpr auto num_fine_values = std::size_t{ 1 } << Rank;

    for (std::size_t n = 0; n != num_fine_values; ++n)
    {
        device_patch_data[dst_patch_base + fine_linear_indices[n]] = value;
    }
}

template <std::size_t Rank>
__global__ void restrict_scalar_patches_kernel(
    double*                                    device_patch_data,
    const intergrid_transfer_task_metadata*    tasks,
    std::size_t                                task_count,
    device_intergrid_transfer_launch_config    config
)
{
    const auto global_idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto total_work = task_count * config.data_flat_size;
    if (global_idx >= total_work)
    {
        return;
    }

    const auto task_idx            = global_idx / config.data_flat_size;
    const auto interior_linear_idx = global_idx % config.data_flat_size;
    const auto task                = tasks[task_idx];

    std::size_t child_patch_offset = 0;
    std::size_t coarse_linear_idx  = 0;
    std::size_t fine_linear_indices[8] = { 0 };
    fill_transfer_mapping<Rank>(
        interior_linear_idx,
        config,
        &child_patch_offset,
        &coarse_linear_idx,
        fine_linear_indices
    );

    const auto src_patch_base = static_cast<std::size_t>(task.source_patch +
                                                         static_cast<std::int32_t>(child_patch_offset)) *
                                config.patch_flat_size;
    const auto dst_patch_base =
        static_cast<std::size_t>(task.destination_patch) * config.patch_flat_size;

    constexpr auto num_fine_values = std::size_t{ 1 } << Rank;

    double sum = 0.0;
    for (std::size_t n = 0; n != num_fine_values; ++n)
    {
        sum += device_patch_data[src_patch_base + fine_linear_indices[n]];
    }

    device_patch_data[dst_patch_base + coarse_linear_idx] =
        sum / static_cast<double>(num_fine_values);
}

auto prepare_transfer_launch(
    std::size_t                              task_count,
    const intergrid_transfer_launch_config&  config,
    device_intergrid_transfer_launch_config* device_config,
    unsigned int*                            blocks
) -> void
{
    if (task_count == 0)
    {
        return;
    }

    device_config->patch_flat_size = config.patch_flat_size;
    device_config->data_flat_size  = config.data_flat_size;
    device_config->halo_width      = config.halo_width;
    for (std::size_t d = 0; d != 3; ++d)
    {
        device_config->padded_strides[d] = config.padded_strides[d];
        device_config->data_sizes[d]     = config.data_sizes[d];
        device_config->data_strides[d]   = config.data_strides[d];
    }

    constexpr unsigned int threads_per_block = 256;
    const auto total_work = task_count * config.data_flat_size;
    *blocks =
        static_cast<unsigned int>((total_work + threads_per_block - 1) / threads_per_block);
}

} // namespace

template <std::size_t Rank>
auto interpolate_scalar_patches_inplace(
    double*                                 device_patch_data,
    const intergrid_transfer_task_metadata* device_tasks,
    std::size_t                             task_count,
    const intergrid_transfer_launch_config& config
) -> void
{
    if (task_count == 0)
    {
        return;
    }

    device_intergrid_transfer_launch_config device_config{};
    unsigned int                           blocks = 0;

    static_assert(Rank >= 1 && Rank <= 3);
    prepare_transfer_launch(task_count, config, &device_config, &blocks);

    constexpr unsigned int threads_per_block = 256;
    interpolate_scalar_patches_kernel<Rank><<<blocks, threads_per_block>>>(
        device_patch_data,
        device_tasks,
        task_count,
        device_config
    );

    throw_if_cuda_error(cudaGetLastError(), "interpolate_scalar_patches_kernel launch");
    throw_if_cuda_error(
        cudaDeviceSynchronize(),
        "interpolate_scalar_patches_kernel synchronize"
    );
}

template <std::size_t Rank>
auto restrict_scalar_patches_inplace(
    double*                                 device_patch_data,
    const intergrid_transfer_task_metadata* device_tasks,
    std::size_t                             task_count,
    const intergrid_transfer_launch_config& config
) -> void
{
    if (task_count == 0)
    {
        return;
    }

    device_intergrid_transfer_launch_config device_config{};
    unsigned int                           blocks = 0;

    static_assert(Rank >= 1 && Rank <= 3);
    prepare_transfer_launch(task_count, config, &device_config, &blocks);

    constexpr unsigned int threads_per_block = 256;
    restrict_scalar_patches_kernel<Rank><<<blocks, threads_per_block>>>(
        device_patch_data,
        device_tasks,
        task_count,
        device_config
    );

    throw_if_cuda_error(cudaGetLastError(), "restrict_scalar_patches_kernel launch");
    throw_if_cuda_error(
        cudaDeviceSynchronize(),
        "restrict_scalar_patches_kernel synchronize"
    );
}

template auto interpolate_scalar_patches_inplace<1>(
    double*,
    const intergrid_transfer_task_metadata*,
    std::size_t,
    const intergrid_transfer_launch_config&
) -> void;

template auto interpolate_scalar_patches_inplace<2>(
    double*,
    const intergrid_transfer_task_metadata*,
    std::size_t,
    const intergrid_transfer_launch_config&
) -> void;

template auto interpolate_scalar_patches_inplace<3>(
    double*,
    const intergrid_transfer_task_metadata*,
    std::size_t,
    const intergrid_transfer_launch_config&
) -> void;

template auto restrict_scalar_patches_inplace<1>(
    double*,
    const intergrid_transfer_task_metadata*,
    std::size_t,
    const intergrid_transfer_launch_config&
) -> void;

template auto restrict_scalar_patches_inplace<2>(
    double*,
    const intergrid_transfer_task_metadata*,
    std::size_t,
    const intergrid_transfer_launch_config&
) -> void;

template auto restrict_scalar_patches_inplace<3>(
    double*,
    const intergrid_transfer_task_metadata*,
    std::size_t,
    const intergrid_transfer_launch_config&
) -> void;

} // namespace amr::cuda

#undef AMR_DEVICE_FORCEINLINE
