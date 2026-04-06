#include "cuda/fvm_refinement_criterion.hpp"

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

__global__ void scalar_patch_amr_kernel(
    const double* packed_patch_data,
    const int*    patch_levels,
    std::size_t   num_patches,
    std::size_t   cells_per_patch,
    double        refine_threshold,
    double        coarsen_threshold,
    int           min_level,
    int           max_level,
    std::int8_t*  decisions
)
{
    const auto patch_idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (patch_idx >= num_patches)
    {
        return;
    }

    const auto patch_base = patch_idx * cells_per_patch;
    double     max_value  = -DBL_MAX;
    for (std::size_t cell_idx = 0; cell_idx != cells_per_patch; ++cell_idx)
    {
        const auto value = packed_patch_data[patch_base + cell_idx];
        max_value        = value > max_value ? value : max_value;
    }

    const auto level = patch_levels[patch_idx];
    if (level < max_level && max_value > refine_threshold)
    {
        decisions[patch_idx] = std::int8_t{ 1 };
        return;
    }

    if (level > min_level && max_value < coarsen_threshold)
    {
        decisions[patch_idx] = std::int8_t{ 2 };
        return;
    }

    decisions[patch_idx] = std::int8_t{ 0 };
}

} // namespace

auto compute_scalar_patch_amr_decisions_from_device(
    const double*                        device_patch_data,
    const int*                           device_patch_levels,
    std::size_t                          level_count,
    const scalar_patch_amr_launch_config& config,
    std::int8_t*                         device_decisions,
    std::size_t                          decision_count
) -> void
{
    if (config.num_patches == 0)
    {
        return;
    }

    if (decision_count < config.num_patches || level_count < config.num_patches)
    {
        throw std::runtime_error("FVM CUDA AMR inputs are smaller than the patch count");
    }

    constexpr unsigned int threads_per_block = 128;
    const auto blocks =
        static_cast<unsigned int>((config.num_patches + threads_per_block - 1) /
                                  threads_per_block);

    scalar_patch_amr_kernel<<<blocks, threads_per_block>>>(
        device_patch_data,
        device_patch_levels,
        config.num_patches,
        config.cells_per_patch,
        config.refine_threshold,
        config.coarsen_threshold,
        config.min_level,
        config.max_level,
        device_decisions
    );

    throw_if_cuda_error(cudaGetLastError(), "scalar_patch_amr_kernel launch");
}

} // namespace amr::cuda
