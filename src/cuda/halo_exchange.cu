#include "cuda/halo_exchange.hpp"

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

struct device_halo_exchange_launch_config
{
    std::size_t num_patches;
    std::size_t patch_flat_size;
    std::size_t components_per_cell;
    std::size_t halo_width;
    std::size_t fanout_1d;
    std::size_t padded_strides[3];
    std::size_t data_sizes[3];
    std::size_t halo_cells_per_dim[3];
    std::size_t halo_work_items_per_patch;
};

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
AMR_DEVICE_FORCEINLINE auto hypercube_offsets(
    const std::size_t* base_coords,
    const std::size_t* strides,
    std::size_t        fanout_1d,
    std::size_t*       out_linear_indices
) -> void
{
    const auto num_entries = fanout_1d * fanout_1d * fanout_1d;
    const auto used_entries =
        Rank == 2 ? (fanout_1d * fanout_1d) : (Rank == 3 ? num_entries : fanout_1d);

    for (std::size_t n = 0; n != used_entries; ++n)
    {
        std::size_t rem = n;
        std::size_t coords[3] = { base_coords[0], base_coords[1], base_coords[2] };
        for (std::size_t d = 0; d != Rank; ++d)
        {
            coords[d] += rem % fanout_1d;
            rem /= fanout_1d;
        }
        out_linear_indices[n] = coords_to_linear<Rank>(coords, strides);
    }
}

template <std::size_t Rank>
AMR_DEVICE_FORCEINLINE auto halo_task_to_coords(
    std::size_t                               halo_cell_idx,
    std::size_t                               direction_index,
    device_halo_exchange_launch_config const& config,
    std::size_t*                              coords
) -> void
{
    const auto dim      = direction_index / 2;
    const auto positive = (direction_index % 2) == 1;

    std::size_t extents[3] = { 1, 1, 1 };
    std::size_t local_strides[3] = { 1, 1, 1 };
    for (std::size_t d = 0; d != Rank; ++d)
    {
        extents[d] = (d == dim) ? config.halo_width : config.data_sizes[d];
    }
    for (std::size_t d = Rank; d-- > 0;)
    {
        if (d + 1 < Rank)
        {
            local_strides[d] = local_strides[d + 1] * extents[d + 1];
        }
    }

    for (std::size_t d = 0; d != Rank; ++d)
    {
        const auto local_coord = halo_cell_idx / local_strides[d];
        halo_cell_idx %= local_strides[d];
        coords[d] = (d == dim)
                        ? ((positive ? (config.halo_width + config.data_sizes[d]) : 0) +
                           local_coord)
                        : (config.halo_width + local_coord);
    }
    for (std::size_t d = Rank; d != 3; ++d)
    {
        coords[d] = 0;
    }
}

template <std::size_t Rank>
AMR_DEVICE_FORCEINLINE auto process_halo_task_from_meta(
    double*                            device_patch_data,
    const halo_direction_metadata&     meta,
    device_halo_exchange_launch_config const& config,
    std::size_t                        patch_idx,
    std::size_t                        direction_index,
    std::size_t                        halo_cell_idx
) -> void
{
    auto* self_patch = device_patch_data + patch_idx * config.patch_flat_size;
    constexpr std::size_t max_hypercube_entries = 8;
    if (meta.relation == halo_neighbor_relation::none)
    {
        return;
    }

    const auto dim      = direction_index / 2;
    const auto positive = (direction_index % 2) == 1;

    std::size_t coords[3] = { 0, 0, 0 };
    halo_task_to_coords<Rank>(halo_cell_idx, direction_index, config, coords);
    const auto linear_idx =
        coords_to_linear<Rank>(coords, config.padded_strides) * config.components_per_cell;

    if (meta.relation == halo_neighbor_relation::same)
    {
        std::size_t from_coords[3] = { coords[0], coords[1], coords[2] };
        from_coords[dim] = positive ? (from_coords[dim] - config.data_sizes[dim])
                                    : (from_coords[dim] + config.data_sizes[dim]);
        const auto from_linear =
            coords_to_linear<Rank>(from_coords, config.padded_strides) *
            config.components_per_cell;
        const auto* other_patch =
            device_patch_data + static_cast<std::size_t>(meta.neighbor) *
                                    config.patch_flat_size;
        for (std::size_t c = 0; c != config.components_per_cell; ++c)
        {
            self_patch[linear_idx + c] = other_patch[from_linear + c];
        }
        return;
    }

    if (meta.relation == halo_neighbor_relation::finer)
    {
        std::size_t base_coords[3] = { coords[0], coords[1], coords[2] };
        base_coords[dim] = positive ? (base_coords[dim] - config.data_sizes[dim])
                                    : (base_coords[dim] + config.data_sizes[dim]);

        std::size_t fine_patch_idx = 0;
        std::size_t stride = 1;
        for (std::size_t d = 0; d != Rank; ++d)
        {
            base_coords[d] =
                ((base_coords[d] - config.halo_width) * config.fanout_1d) %
                    config.data_sizes[d] +
                config.halo_width;
            if (d != dim)
            {
                const auto section_size = config.data_sizes[d] / config.fanout_1d;
                const auto section_idx =
                    (coords[d] - config.halo_width) / section_size;
                fine_patch_idx += section_idx * stride;
                stride *= config.fanout_1d;
            }
        }

        std::size_t fine_linear_indices[max_hypercube_entries] = { 0 };
        hypercube_offsets<Rank>(
            base_coords,
            config.padded_strides,
            config.fanout_1d,
            fine_linear_indices
        );

        const auto num_fine_values =
            Rank == 1 ? config.fanout_1d
                      : (Rank == 2 ? (config.fanout_1d * config.fanout_1d)
                                   : (config.fanout_1d * config.fanout_1d *
                                      config.fanout_1d));
        const auto* fine_patch =
            device_patch_data +
            static_cast<std::size_t>(meta.finer_ids[fine_patch_idx]) *
                config.patch_flat_size;

        for (std::size_t c = 0; c != config.components_per_cell; ++c)
        {
            double sum = 0.0;
            for (std::size_t n = 0; n != num_fine_values; ++n)
            {
                sum += fine_patch[fine_linear_indices[n] * config.components_per_cell + c];
            }
            self_patch[linear_idx + c] = sum / static_cast<double>(num_fine_values);
        }
        return;
    }

    if (meta.relation == halo_neighbor_relation::coarser)
    {
        std::size_t from_coords[3] = { coords[0], coords[1], coords[2] };
        from_coords[dim] = positive ? (from_coords[dim] - config.data_sizes[dim])
                                    : (from_coords[dim] + config.data_sizes[dim]);
        for (std::size_t d = 0; d != Rank; ++d)
        {
            const auto cells_per_block = config.data_sizes[d] / config.fanout_1d;
            const auto fine_mapped_idx = from_coords[d] - config.halo_width;
            from_coords[d] = config.halo_width +
                             static_cast<std::size_t>(meta.quadrant[d]) * cells_per_block +
                             fine_mapped_idx / config.fanout_1d;
        }

        const auto from_linear =
            coords_to_linear<Rank>(from_coords, config.padded_strides) *
            config.components_per_cell;
        const auto* coarse_patch =
            device_patch_data + static_cast<std::size_t>(meta.neighbor) *
                                    config.patch_flat_size;
        for (std::size_t c = 0; c != config.components_per_cell; ++c)
        {
            self_patch[linear_idx + c] = coarse_patch[from_linear + c];
        }
    }
}

template <std::size_t Rank>
AMR_DEVICE_FORCEINLINE auto decode_fused_halo_work_item(
    std::size_t local_work_item_idx,
    device_halo_exchange_launch_config const& config,
    std::size_t& direction_index,
    std::size_t& halo_cell_idx
) -> void
{
    for (std::size_t dim = 0; dim != Rank; ++dim)
    {
        const auto halo_cells_per_patch_dim = config.halo_cells_per_dim[dim];
        const auto dim_work_items = 2 * halo_cells_per_patch_dim;
        if (local_work_item_idx < dim_work_items)
        {
            direction_index = 2 * dim + (local_work_item_idx >= halo_cells_per_patch_dim);
            halo_cell_idx   = local_work_item_idx % halo_cells_per_patch_dim;
            return;
        }
        local_work_item_idx -= dim_work_items;
    }

    direction_index = 0;
    halo_cell_idx   = 0;
}

template <std::size_t Rank>
AMR_DEVICE_FORCEINLINE auto process_halo_task(
    double*                            device_patch_data,
    const halo_direction_metadata*     neighbor_metadata,
    device_halo_exchange_launch_config const& config,
    std::size_t                        patch_idx,
    std::size_t                        direction_index,
    std::size_t                        halo_cell_idx
) -> void
{
    const auto& meta = neighbor_metadata[patch_idx * (2 * Rank) + direction_index];
    process_halo_task_from_meta<Rank>(
        device_patch_data,
        meta,
        config,
        patch_idx,
        direction_index,
        halo_cell_idx
    );
}

template <std::size_t Rank>
__global__ void halo_exchange_scalar_kernel_fused(
    double*                                    device_patch_data,
    const halo_direction_metadata*             neighbor_metadata,
    device_halo_exchange_launch_config         config
)
{
    const auto global_idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto total_work_items = config.num_patches * config.halo_work_items_per_patch;
    if (global_idx >= total_work_items)
    {
        return;
    }

    const auto patch_idx = global_idx / config.halo_work_items_per_patch;
    const auto local_work_item_idx = global_idx % config.halo_work_items_per_patch;

    std::size_t direction_index = 0;
    std::size_t halo_cell_idx   = 0;
    decode_fused_halo_work_item<Rank>(
        local_work_item_idx, config, direction_index, halo_cell_idx
    );

    process_halo_task<Rank>(
        device_patch_data,
        neighbor_metadata,
        config,
        patch_idx,
        direction_index,
        halo_cell_idx
    );
}

template <std::size_t Rank>
auto launch_halo_exchange_fused_kernel(
    double*                                   device_patch_data,
    const halo_direction_metadata*            device_neighbor_metadata,
    device_halo_exchange_launch_config const& device_config
) -> void
{
    const auto total_work_items =
        device_config.num_patches * device_config.halo_work_items_per_patch;
    if (total_work_items == 0)
    {
        return;
    }

    constexpr unsigned int threads_per_block = 256;
    const auto blocks = static_cast<unsigned int>(
        (total_work_items + threads_per_block - 1) / threads_per_block
    );

    halo_exchange_scalar_kernel_fused<Rank><<<blocks, threads_per_block>>>(
        device_patch_data,
        device_neighbor_metadata,
        device_config
    );
}

} // namespace

auto halo_exchange_scalar_patches_inplace(
    double*                            device_patch_data,
    const halo_direction_metadata*     device_neighbor_metadata,
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

    device_halo_exchange_launch_config device_config{};
    device_config.num_patches    = config.num_patches;
    device_config.patch_flat_size = config.patch_flat_size;
    device_config.components_per_cell = config.components_per_cell;
    device_config.halo_width     = config.halo_width;
    device_config.fanout_1d      = config.fanout_1d;
    for (std::size_t d = 0; d != 3; ++d)
    {
        device_config.padded_strides[d] = config.padded_strides[d];
        device_config.data_sizes[d] = config.data_sizes[d];
        device_config.halo_cells_per_dim[d] = config.halo_cells_per_dim[d];
    }
    device_config.halo_work_items_per_patch = config.halo_work_items_per_patch;

    switch (config.rank)
    {
    case 1:
        launch_halo_exchange_fused_kernel<1>(
            device_patch_data, device_neighbor_metadata, device_config
        );
        break;
    case 2:
        launch_halo_exchange_fused_kernel<2>(
            device_patch_data, device_neighbor_metadata, device_config
        );
        break;
    default:
        launch_halo_exchange_fused_kernel<3>(
            device_patch_data, device_neighbor_metadata, device_config
        );
        break;
    }

    throw_if_cuda_error(cudaGetLastError(), "halo_exchange_scalar_kernel launch");
}

} // namespace amr::cuda

#undef AMR_DEVICE_FORCEINLINE
