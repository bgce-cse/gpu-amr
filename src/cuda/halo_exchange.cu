#include "cuda/halo_exchange.hpp"

#include <cuda_runtime.h>

#include <cfloat>
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

struct device_halo_exchange_launch_config
{
    std::size_t num_patches;
    std::size_t patch_flat_size;
    std::size_t rank;
    std::size_t halo_width;
    std::size_t fanout_1d;
    std::size_t padded_sizes[3];
    std::size_t padded_strides[3];
    std::size_t data_sizes[3];
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

__device__ auto is_direction_halo_cell(
    const std::size_t* coords,
    std::size_t        rank,
    std::size_t        dim,
    bool               positive,
    std::size_t        halo_width,
    const std::size_t* data_sizes
) -> bool
{
    for (std::size_t d = 0; d != rank; ++d)
    {
        if (d == dim)
        {
            const auto low = positive ? halo_width + data_sizes[d] : 0;
            const auto high = positive ? halo_width + data_sizes[d] + halo_width : halo_width;
            if (coords[d] < low || coords[d] >= high)
            {
                return false;
            }
        }
        else if (coords[d] < halo_width || coords[d] >= halo_width + data_sizes[d])
        {
            return false;
        }
    }
    return true;
}

__device__ auto hypercube_offsets(
    const std::size_t* base_coords,
    const std::size_t* strides,
    std::size_t        rank,
    std::size_t        fanout_1d,
    std::size_t*       out_linear_indices
) -> void
{
    const auto num_entries = fanout_1d * fanout_1d * fanout_1d;
    const auto used_entries =
        rank == 2 ? (fanout_1d * fanout_1d) : (rank == 3 ? num_entries : fanout_1d);

    for (std::size_t n = 0; n != used_entries; ++n)
    {
        std::size_t rem = n;
        std::size_t coords[3] = { base_coords[0], base_coords[1], base_coords[2] };
        for (std::size_t d = 0; d != rank; ++d)
        {
            coords[d] += rem % fanout_1d;
            rem /= fanout_1d;
        }
        out_linear_indices[n] = coords_to_linear(coords, strides, rank);
    }
}

__global__ void halo_exchange_scalar_kernel(
    double*                                    device_patch_data,
    const halo_direction_metadata*             neighbor_metadata,
    device_halo_exchange_launch_config         config
)
{
    const auto patch_idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (patch_idx >= config.num_patches)
    {
        return;
    }

    auto* self_patch = device_patch_data + patch_idx * config.patch_flat_size;
    constexpr std::size_t max_hypercube_entries = 8;

    for (std::size_t dir_index = 0; dir_index != 2 * config.rank; ++dir_index)
    {
        const auto& meta = neighbor_metadata[patch_idx * (2 * config.rank) + dir_index];
        if (meta.relation == halo_neighbor_relation::none)
        {
            continue;
        }

        const auto dim      = dir_index / 2;
        const auto positive = (dir_index % 2) == 1;

        for (std::size_t linear_idx = 0; linear_idx != config.patch_flat_size; ++linear_idx)
        {
            std::size_t coords[3] = { 0, 0, 0 };
            linear_to_coords(
                linear_idx,
                config.padded_strides,
                config.padded_sizes,
                config.rank,
                coords
            );

            if (!is_direction_halo_cell(
                    coords,
                    config.rank,
                    dim,
                    positive,
                    config.halo_width,
                    config.data_sizes
                ))
            {
                continue;
            }

            if (meta.relation == halo_neighbor_relation::same)
            {
                std::size_t from_coords[3] = { coords[0], coords[1], coords[2] };
                from_coords[dim] = positive ? (from_coords[dim] - config.data_sizes[dim])
                                            : (from_coords[dim] + config.data_sizes[dim]);
                const auto from_linear =
                    coords_to_linear(from_coords, config.padded_strides, config.rank);
                const auto* other_patch =
                    device_patch_data + static_cast<std::size_t>(meta.neighbor) *
                                            config.patch_flat_size;
                self_patch[linear_idx] = other_patch[from_linear];
                continue;
            }

            if (meta.relation == halo_neighbor_relation::finer)
            {
                std::size_t base_coords[3] = { coords[0], coords[1], coords[2] };
                base_coords[dim] = positive ? (base_coords[dim] - config.data_sizes[dim])
                                            : (base_coords[dim] + config.data_sizes[dim]);

                std::size_t fine_patch_idx = 0;
                std::size_t stride = 1;
                for (std::size_t d = 0; d != config.rank; ++d)
                {
                    base_coords[d] =
                        ((base_coords[d] - config.halo_width) * config.fanout_1d) %
                            config.data_sizes[d] +
                        config.halo_width;
                    if (d != dim)
                    {
                        const auto section_size =
                            config.data_sizes[d] / config.fanout_1d;
                        const auto section_idx =
                            (coords[d] - config.halo_width) / section_size;
                        fine_patch_idx += section_idx * stride;
                        stride *= config.fanout_1d;
                    }
                }

                std::size_t fine_linear_indices[max_hypercube_entries] = { 0 };
                hypercube_offsets(
                    base_coords,
                    config.padded_strides,
                    config.rank,
                    config.fanout_1d,
                    fine_linear_indices
                );

                const auto num_fine_values =
                    config.rank == 2 ? (config.fanout_1d * config.fanout_1d)
                                     : (config.fanout_1d * config.fanout_1d * config.fanout_1d);
                const auto* fine_patch =
                    device_patch_data +
                    static_cast<std::size_t>(meta.finer_ids[fine_patch_idx]) *
                        config.patch_flat_size;

                double sum = 0.0;
                for (std::size_t n = 0; n != num_fine_values; ++n)
                {
                    sum += fine_patch[fine_linear_indices[n]];
                }
                self_patch[linear_idx] = sum / static_cast<double>(num_fine_values);
                continue;
            }

            if (meta.relation == halo_neighbor_relation::coarser)
            {
                std::size_t from_coords[3] = { coords[0], coords[1], coords[2] };
                from_coords[dim] = positive ? (from_coords[dim] - config.data_sizes[dim])
                                            : (from_coords[dim] + config.data_sizes[dim]);
                for (std::size_t d = 0; d != config.rank; ++d)
                {
                    const auto cells_per_block = config.data_sizes[d] / config.fanout_1d;
                    from_coords[d] = config.halo_width +
                                     static_cast<std::size_t>(meta.quadrant[d]) * cells_per_block +
                                     (from_coords[d] - config.halo_width) / config.fanout_1d;
                }

                const auto from_linear =
                    coords_to_linear(from_coords, config.padded_strides, config.rank);
                const auto* coarse_patch =
                    device_patch_data + static_cast<std::size_t>(meta.neighbor) *
                                            config.patch_flat_size;
                self_patch[linear_idx] = coarse_patch[from_linear];
            }
        }
    }
}

} // namespace

auto halo_exchange_scalar_patches_inplace(
    double*                            device_patch_data,
    const halo_direction_metadata*     host_neighbor_metadata,
    std::size_t                        metadata_count,
    const halo_exchange_launch_config& config
) -> void
{
    if (config.num_patches == 0)
    {
        return;
    }

    const auto expected_metadata_count = config.num_patches * 2 * config.rank;
    if (metadata_count != expected_metadata_count)
    {
        throw std::runtime_error("Halo metadata size does not match launch configuration");
    }

    halo_direction_metadata* device_neighbor_metadata = nullptr;
    throw_if_cuda_error(
        cudaMalloc(
            reinterpret_cast<void**>(&device_neighbor_metadata),
            metadata_count * sizeof(halo_direction_metadata)
        ),
        "cudaMalloc(device_neighbor_metadata)"
    );

    try
    {
        throw_if_cuda_error(
            cudaMemcpy(
                device_neighbor_metadata,
                host_neighbor_metadata,
                metadata_count * sizeof(halo_direction_metadata),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy(host_to_device halo metadata)"
        );

        device_halo_exchange_launch_config device_config{};
        device_config.num_patches    = config.num_patches;
        device_config.patch_flat_size = config.patch_flat_size;
        device_config.rank           = config.rank;
        device_config.halo_width     = config.halo_width;
        device_config.fanout_1d      = config.fanout_1d;
        for (std::size_t d = 0; d != 3; ++d)
        {
            device_config.padded_sizes[d] = config.padded_sizes[d];
            device_config.padded_strides[d] = config.padded_strides[d];
            device_config.data_sizes[d] = config.data_sizes[d];
        }

        constexpr unsigned int threads_per_block = 128;
        const auto blocks =
            static_cast<unsigned int>((config.num_patches + threads_per_block - 1) /
                                      threads_per_block);

        halo_exchange_scalar_kernel<<<blocks, threads_per_block>>>(
            device_patch_data,
            device_neighbor_metadata,
            device_config
        );

        throw_if_cuda_error(cudaGetLastError(), "halo_exchange_scalar_kernel launch");
        throw_if_cuda_error(cudaDeviceSynchronize(), "halo_exchange_scalar_kernel synchronize");
    }
    catch (...)
    {
        cudaFree(device_neighbor_metadata);
        throw;
    }

    cudaFree(device_neighbor_metadata);
}

} // namespace amr::cuda
