#include "cuda/permutation.hpp"
#include "cuda/device_buffer.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <vector>

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
    std::size_t*                pinned_sources       = nullptr;
    std::vector<unsigned char*> alternate_buffers{};
    std::vector<std::size_t>    alternate_capacities{};
    std::size_t*                device_sources       = nullptr;
    std::size_t                 sources_capacity     = 0;
    void*                       sources_copy_fence   = nullptr;
    std::size_t                 copy_pending         = 0;

    ~permutation_scratch_buffers()
    {
        if (copy_pending != 0)
        {
            amr::cuda::async_copy_fence_wait(sources_copy_fence);
        }
        amr::cuda::async_copy_fence_destroy(sources_copy_fence);
        amr::cuda::host_pinned_free(static_cast<void*>(pinned_sources));
        (void)cudaFree(device_sources);
        for (auto* buffer : alternate_buffers)
        {
            (void)cudaFree(buffer);
        }
    }

    auto ensure_alternate_buffer_count(std::size_t required_count) -> void
    {
        if (required_count <= alternate_buffers.size())
        {
            return;
        }

        alternate_buffers.resize(required_count, nullptr);
        alternate_capacities.resize(required_count, 0);
    }

    auto ensure_alternate_capacity(std::size_t buffer_index, std::size_t required_bytes)
        -> void
    {
        ensure_alternate_buffer_count(buffer_index + 1);
        if (required_bytes <= alternate_capacities[buffer_index])
        {
            return;
        }

        const auto grown_capacity =
            std::max(required_bytes * 2, std::size_t{ 256 * 1024 });

        (void)cudaFree(alternate_buffers[buffer_index]);
        alternate_buffers[buffer_index] = nullptr;
        throw_if_cuda_error(
            cudaMalloc(
                reinterpret_cast<void**>(&alternate_buffers[buffer_index]),
                grown_capacity
            ),
            "cudaMalloc(alternate_permutation_buffer)"
        );
        alternate_capacities[buffer_index] = grown_capacity;
    }

    auto ensure_sources_capacity(std::size_t required_count) -> void
    {
        if (required_count <= sources_capacity)
        {
            return;
        }

        const auto grown_capacity =
            std::max(required_count * 2, std::size_t{ 4096 });

        (void)cudaFree(device_sources);
        device_sources = nullptr;
        throw_if_cuda_error(
            cudaMalloc(
                reinterpret_cast<void**>(&device_sources),
                grown_capacity * sizeof(std::size_t)
            ),
            "cudaMalloc(device_sources)"
        );
        amr::cuda::host_pinned_free(static_cast<void*>(pinned_sources));
        pinned_sources = static_cast<std::size_t*>(
            amr::cuda::host_pinned_malloc(grown_capacity * sizeof(std::size_t))
        );
        if (sources_copy_fence == nullptr)
        {
            sources_copy_fence = amr::cuda::async_copy_fence_create();
        }
        sources_capacity = grown_capacity;
    }

    auto wait_for_pending_sources_copy() -> void
    {
        if (copy_pending != 0)
        {
            amr::cuda::async_copy_fence_wait(sources_copy_fence);
            copy_pending = 0;
        }
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
    unsigned char*     source_bytes,
    unsigned char*     destination_bytes,
    std::size_t        patch_count,
    std::size_t        patch_bytes,
    const std::size_t* device_sources
) -> unsigned char*
{
    const auto total_bytes = patch_count * patch_bytes;

    constexpr unsigned int threads_per_block = 256;
    const auto blocks =
        static_cast<unsigned int>((total_bytes + threads_per_block - 1) /
                                  threads_per_block);

    permute_patches_kernel<<<blocks, threads_per_block>>>(
        source_bytes,
        destination_bytes,
        device_sources,
        patch_count,
        patch_bytes
    );

    throw_if_cuda_error(cudaGetLastError(), "permute_patches_kernel launch");
    return source_bytes;
}

} // namespace

auto permute_patches_inplace_batch(
    std::span<void**>            device_buffer_slots,
    std::span<const std::size_t> patch_bytes,
    std::size_t                  patch_count,
    const std::size_t*           host_sources,
    std::size_t                  source_count
) -> void
{
    if (patch_count == 0 || device_buffer_slots.empty())
    {
        return;
    }

    if (device_buffer_slots.size() != patch_bytes.size())
    {
        throw std::runtime_error("Permutation buffer count does not match patch byte count");
    }

    validate_permutation_size(patch_count, source_count);

    auto& scratch = permutation_scratch();
    auto  lock    = std::lock_guard<std::mutex>(permutation_scratch_mutex());

    scratch.ensure_sources_capacity(patch_count);
    scratch.ensure_alternate_buffer_count(device_buffer_slots.size());
    scratch.wait_for_pending_sources_copy();

    for (std::size_t i = 0; i < patch_count; ++i)
    {
        scratch.pinned_sources[i] = host_sources[i];
    }

    amr::cuda::copy_host_to_device_async(
        static_cast<void*>(scratch.device_sources),
        static_cast<void const*>(scratch.pinned_sources),
        patch_count * sizeof(std::size_t)
    );
    amr::cuda::async_copy_fence_record(scratch.sources_copy_fence);
    scratch.copy_pending = 1;

    for (std::size_t i = 0; i < device_buffer_slots.size(); ++i)
    {
        auto* source_bytes      = static_cast<unsigned char*>(*device_buffer_slots[i]);
        const auto total_bytes  = patch_count * patch_bytes[i];

        scratch.ensure_alternate_capacity(i, total_bytes);
        auto* destination_bytes = scratch.alternate_buffers[i];

        scratch.alternate_buffers[i] = permute_patches_with_device_sources(
            source_bytes,
            destination_bytes,
            patch_count,
            patch_bytes[i],
            scratch.device_sources
        );
        *device_buffer_slots[i] = destination_bytes;
    }
}

} // namespace amr::cuda
