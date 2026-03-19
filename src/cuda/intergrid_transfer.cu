#include "cuda/intergrid_transfer.hpp"

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>

namespace amr::cuda
{

namespace
{

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
    std::size_t rank;
    std::size_t halo_width;
    std::size_t fanout_1d;
    std::size_t padded_strides[3];
    std::size_t data_sizes[3];
    std::size_t data_strides[3];
};

__device__ auto linear_to_coords(
    std::size_t        linear_idx,
    const std::size_t* strides,
    const std::size_t* sizes,
    std::size_t        rank,
    std::size_t*       coords
) -> void
{
    for (std::size_t d = 0; d != rank; ++d)
    {
        coords[d] = linear_idx / strides[d];
        linear_idx %= strides[d];
        if (coords[d] >= sizes[d])
        {
            coords[d] = 0;
        }
    }
}

__device__ auto coords_to_linear(
    const std::size_t* coords,
    const std::size_t* strides,
    std::size_t        rank
) -> std::size_t
{
    std::size_t linear = 0;
    for (std::size_t d = 0; d != rank; ++d)
    {
        linear += coords[d] * strides[d];
    }
    return linear;
}

__device__ auto fill_transfer_mapping(
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

    linear_to_coords(
        interior_linear_idx,
        config.data_strides,
        config.data_sizes,
        config.rank,
        coarse_coords
    );

    *fine_patch_idx = 0;
    for (std::size_t d = 0; d != config.rank; ++d)
    {
        padded_coords[d] = coarse_coords[d] + config.halo_width;

        const auto section_size = config.data_sizes[d] / config.fanout_1d;
        const auto child_coord  = coarse_coords[d] / section_size;
        *fine_patch_idx = (*fine_patch_idx * config.fanout_1d) + child_coord;

        base_fine_coords[d] =
            (coarse_coords[d] % section_size) * config.fanout_1d + config.halo_width;
    }

    *coarse_linear_idx =
        coords_to_linear(padded_coords, config.padded_strides, config.rank);

    const auto fanout = config.fanout_1d;
    const auto num_fine_values =
        config.rank == 1 ? fanout
                         : (config.rank == 2 ? fanout * fanout : fanout * fanout * fanout);

    for (std::size_t n = 0; n != num_fine_values; ++n)
    {
        std::size_t rem            = n;
        std::size_t fine_coords[3] = {
            base_fine_coords[0], base_fine_coords[1], base_fine_coords[2]
        };

        std::size_t stride = num_fine_values;
        for (std::size_t d = 0; d != config.rank; ++d)
        {
            stride /= fanout;
            fine_coords[d] += rem / stride;
            rem %= stride;
        }

        fine_linear_indices[n] =
            coords_to_linear(fine_coords, config.padded_strides, config.rank);
    }
}

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
    fill_transfer_mapping(
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

    const auto fanout = config.fanout_1d;
    const auto num_fine_values =
        config.rank == 1 ? fanout
                         : (config.rank == 2 ? fanout * fanout : fanout * fanout * fanout);

    for (std::size_t n = 0; n != num_fine_values; ++n)
    {
        device_patch_data[dst_patch_base + fine_linear_indices[n]] = value;
    }
}

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
    fill_transfer_mapping(
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

    const auto fanout = config.fanout_1d;
    const auto num_fine_values =
        config.rank == 1 ? fanout
                         : (config.rank == 2 ? fanout * fanout : fanout * fanout * fanout);

    double sum = 0.0;
    for (std::size_t n = 0; n != num_fine_values; ++n)
    {
        sum += device_patch_data[src_patch_base + fine_linear_indices[n]];
    }

    device_patch_data[dst_patch_base + coarse_linear_idx] =
        sum / static_cast<double>(num_fine_values);
}

auto prepare_transfer_launch(
    const intergrid_transfer_task_metadata* host_tasks,
    std::size_t                             task_count,
    const intergrid_transfer_launch_config& config,
    intergrid_transfer_task_metadata**      device_tasks,
    device_intergrid_transfer_launch_config* device_config,
    unsigned int*                           blocks
) -> void
{
    if (task_count == 0)
    {
        return;
    }

    throw_if_cuda_error(
        cudaMalloc(
            reinterpret_cast<void**>(device_tasks),
            task_count * sizeof(intergrid_transfer_task_metadata)
        ),
        "cudaMalloc(device intergrid tasks)"
    );

    try
    {
        throw_if_cuda_error(
            cudaMemcpy(
                *device_tasks,
                host_tasks,
                task_count * sizeof(intergrid_transfer_task_metadata),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy(host_to_device intergrid tasks)"
        );

        device_config->patch_flat_size = config.patch_flat_size;
        device_config->data_flat_size  = config.data_flat_size;
        device_config->rank            = config.rank;
        device_config->halo_width      = config.halo_width;
        device_config->fanout_1d       = config.fanout_1d;
        for (std::size_t d = 0; d != 3; ++d)
        {
            device_config->padded_strides[d] = config.padded_strides[d];
            device_config->data_sizes[d]     = config.data_sizes[d];
            device_config->data_strides[d]   = config.data_strides[d];
        }

        constexpr unsigned int threads_per_block = 256;
        const auto total_work = task_count * config.data_flat_size;
        *blocks =
            static_cast<unsigned int>((total_work + threads_per_block - 1) /
                                      threads_per_block);
    }
    catch (...)
    {
        cudaFree(*device_tasks);
        throw;
    }
}

} // namespace

auto interpolate_scalar_patches_inplace(
    double*                                 device_patch_data,
    const intergrid_transfer_task_metadata* host_tasks,
    std::size_t                             task_count,
    const intergrid_transfer_launch_config& config
) -> void
{
    if (task_count == 0)
    {
        return;
    }

    intergrid_transfer_task_metadata*      device_tasks = nullptr;
    device_intergrid_transfer_launch_config device_config{};
    unsigned int                           blocks = 0;

    try
    {
        prepare_transfer_launch(
            host_tasks,
            task_count,
            config,
            &device_tasks,
            &device_config,
            &blocks
        );

        constexpr unsigned int threads_per_block = 256;
        interpolate_scalar_patches_kernel<<<blocks, threads_per_block>>>(
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
    catch (...)
    {
        cudaFree(device_tasks);
        throw;
    }

    cudaFree(device_tasks);
}

auto restrict_scalar_patches_inplace(
    double*                                 device_patch_data,
    const intergrid_transfer_task_metadata* host_tasks,
    std::size_t                             task_count,
    const intergrid_transfer_launch_config& config
) -> void
{
    if (task_count == 0)
    {
        return;
    }

    intergrid_transfer_task_metadata*      device_tasks = nullptr;
    device_intergrid_transfer_launch_config device_config{};
    unsigned int                           blocks = 0;

    try
    {
        prepare_transfer_launch(
            host_tasks,
            task_count,
            config,
            &device_tasks,
            &device_config,
            &blocks
        );

        constexpr unsigned int threads_per_block = 256;
        restrict_scalar_patches_kernel<<<blocks, threads_per_block>>>(
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
    catch (...)
    {
        cudaFree(device_tasks);
        throw;
    }

    cudaFree(device_tasks);
}

} // namespace amr::cuda
