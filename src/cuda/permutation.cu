#include "cuda/permutation.hpp"

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

__global__ void permute_patches_kernel(
    const unsigned char* src,
    unsigned char*       dst,
    const std::size_t*   sources,
    std::size_t          patch_count,
    std::size_t          patch_bytes
)
{
    const auto global_idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto total_bytes = patch_count * patch_bytes;
    if (global_idx >= total_bytes)
    {
        return;
    }

    const auto dst_patch = global_idx / patch_bytes;
    const auto byte_idx  = global_idx % patch_bytes;
    const auto src_patch = sources[dst_patch];

    dst[global_idx] = src[src_patch * patch_bytes + byte_idx];
}

} // namespace

auto permute_patches_inplace(
    void*              device_buffer,
    std::size_t        patch_count,
    std::size_t        patch_bytes,
    const std::size_t* host_sources,
    std::size_t        source_count
) -> void
{
    if (patch_count == 0)
    {
        return;
    }

    if (source_count != patch_count)
    {
        throw std::runtime_error("Permutation source count does not match patch count");
    }

    auto* device_bytes = static_cast<unsigned char*>(device_buffer);
    unsigned char* temp_buffer    = nullptr;
    std::size_t*   device_sources = nullptr;

    throw_if_cuda_error(
        cudaMalloc(reinterpret_cast<void**>(&temp_buffer), patch_count * patch_bytes),
        "cudaMalloc(temp_buffer)"
    );

    try
    {
        throw_if_cuda_error(
            cudaMalloc(reinterpret_cast<void**>(&device_sources), patch_count * sizeof(std::size_t)),
            "cudaMalloc(device_sources)"
        );

        throw_if_cuda_error(
            cudaMemcpy(
                device_sources,
                host_sources,
                patch_count * sizeof(std::size_t),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy(host_to_device sources)"
        );

        constexpr unsigned int threads_per_block = 256;
        const auto total_bytes = patch_count * patch_bytes;
        const auto blocks =
            static_cast<unsigned int>((total_bytes + threads_per_block - 1) /
                                      threads_per_block);

        permute_patches_kernel<<<blocks, threads_per_block>>>(
            device_bytes, temp_buffer, device_sources, patch_count, patch_bytes
        );

        throw_if_cuda_error(cudaGetLastError(), "permute_patches_kernel launch");
        throw_if_cuda_error(cudaDeviceSynchronize(), "permute_patches_kernel synchronize");

        throw_if_cuda_error(
            cudaMemcpy(
                device_bytes,
                temp_buffer,
                patch_count * patch_bytes,
                cudaMemcpyDeviceToDevice
            ),
            "cudaMemcpy(device_to_device permuted patches)"
        );
    }
    catch (...)
    {
        cudaFree(device_sources);
        cudaFree(temp_buffer);
        throw;
    }

    cudaFree(device_sources);
    cudaFree(temp_buffer);
}

} // namespace amr::cuda
