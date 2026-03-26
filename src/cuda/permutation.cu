#include "cuda/permutation.hpp"

#include <cuda_runtime.h>

#include <mutex>
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

struct permutation_scratch_buffers
{
    unsigned char* temp_buffer         = nullptr;
    std::size_t    temp_capacity_bytes = 0;
    std::size_t*   device_sources      = nullptr;
    std::size_t    sources_capacity    = 0;

    ~permutation_scratch_buffers()
    {
        (void)cudaFree(device_sources);
        (void)cudaFree(temp_buffer);
    }

    auto ensure_temp_capacity(std::size_t required_bytes) -> void
    {
        if (required_bytes <= temp_capacity_bytes)
        {
            return;
        }

        (void)cudaFree(temp_buffer);
        temp_buffer = nullptr;
        throw_if_cuda_error(
            cudaMalloc(reinterpret_cast<void**>(&temp_buffer), required_bytes),
            "cudaMalloc(temp_buffer)"
        );
        temp_capacity_bytes = required_bytes;
    }

    auto ensure_sources_capacity(std::size_t required_count) -> void
    {
        if (required_count <= sources_capacity)
        {
            return;
        }

        (void)cudaFree(device_sources);
        device_sources = nullptr;
        throw_if_cuda_error(
            cudaMalloc(
                reinterpret_cast<void**>(&device_sources),
                required_count * sizeof(std::size_t)
            ),
            "cudaMalloc(device_sources)"
        );
        sources_capacity = required_count;
    }
};

auto permutation_scratch() -> permutation_scratch_buffers&
{
    static permutation_scratch_buffers scratch;
    return scratch;
}

auto permutation_scratch_mutex() -> std::mutex&
{
    static std::mutex mutex;
    return mutex;
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

auto validate_permutation_size(std::size_t patch_count, std::size_t source_count)
    -> void
{
    if (source_count != patch_count)
    {
        throw std::runtime_error("Permutation source count does not match patch count");
    }
}

auto permute_patches_with_device_sources(
    unsigned char*     device_bytes,
    std::size_t        patch_count,
    std::size_t        patch_bytes,
    const std::size_t* device_sources,
    unsigned char*     temp_buffer
) -> void
{
    const auto total_bytes = patch_count * patch_bytes;

    constexpr unsigned int threads_per_block = 256;
    const auto blocks =
        static_cast<unsigned int>((total_bytes + threads_per_block - 1) /
                                  threads_per_block);

    permute_patches_kernel<<<blocks, threads_per_block>>>(
        device_bytes,
        temp_buffer,
        device_sources,
        patch_count,
        patch_bytes
    );

    throw_if_cuda_error(cudaGetLastError(), "permute_patches_kernel launch");

    throw_if_cuda_error(
        cudaMemcpy(
            device_bytes,
            temp_buffer,
            total_bytes,
            cudaMemcpyDeviceToDevice
        ),
        "cudaMemcpy(device_to_device permuted patches)"
    );
}

} // namespace

auto permute_patches_inplace_batch(
    std::span<void*>             device_buffers,
    std::span<const std::size_t> patch_bytes,
    std::size_t                  patch_count,
    const std::size_t*           host_sources,
    std::size_t                  source_count
) -> void
{
    if (patch_count == 0 || device_buffers.empty())
    {
        return;
    }

    if (device_buffers.size() != patch_bytes.size())
    {
        throw std::runtime_error("Permutation buffer count does not match patch byte count");
    }

    validate_permutation_size(patch_count, source_count);

    auto& scratch = permutation_scratch();
    auto  lock    = std::lock_guard<std::mutex>(permutation_scratch_mutex());

    scratch.ensure_sources_capacity(patch_count);

    throw_if_cuda_error(
        cudaMemcpy(
            scratch.device_sources,
            host_sources,
            patch_count * sizeof(std::size_t),
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy(host_to_device sources)"
    );

    for (std::size_t i = 0; i < device_buffers.size(); ++i)
    {
        auto* device_bytes      = static_cast<unsigned char*>(device_buffers[i]);
        const auto total_bytes  = patch_count * patch_bytes[i];

        scratch.ensure_temp_capacity(total_bytes);

        permute_patches_with_device_sources(
            device_bytes,
            patch_count,
            patch_bytes[i],
            scratch.device_sources,
            scratch.temp_buffer
        );
    }
}

auto permute_patches_inplace(
    void*              device_buffer,
    std::size_t        patch_count,
    std::size_t        patch_bytes,
    const std::size_t* host_sources,
    std::size_t        source_count
) -> void
{
    auto device_buffers = std::span{ &device_buffer, std::size_t{ 1 } };
    auto patch_sizes    = std::span{ &patch_bytes, std::size_t{ 1 } };

    permute_patches_inplace_batch(
        device_buffers,
        patch_sizes,
        patch_count,
        host_sources,
        source_count
    );
}

} // namespace amr::cuda
