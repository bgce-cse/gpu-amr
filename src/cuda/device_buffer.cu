#include "cuda/device_buffer.hpp"
#include "cuda/profiler.hpp"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

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

} // namespace

auto device_malloc(std::size_t bytes) -> void*
{
    void* ptr = nullptr;
    throw_if_cuda_error(cudaMalloc(&ptr, bytes), "cudaMalloc");
    return ptr;
}

auto device_free(void* ptr) noexcept -> void
{
    if (ptr == nullptr)
    {
        return;
    }

    (void)cudaFree(ptr);
}

auto host_pinned_malloc(std::size_t bytes) -> void*
{
    void* ptr = nullptr;
    throw_if_cuda_error(cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault), "cudaHostAlloc");
    return ptr;
}

auto host_pinned_free(void* ptr) noexcept -> void
{
    if (ptr == nullptr)
    {
        return;
    }

    (void)cudaFreeHost(ptr);
}

auto copy_host_to_device(void* dst, const void* src, std::size_t bytes) -> void
{
    if (bytes == 0)
    {
        return;
    }

    throw_if_cuda_error(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D");
}

auto copy_host_to_device_async(void* dst, const void* src, std::size_t bytes) -> void
{
    if (bytes == 0)
    {
        return;
    }

    throw_if_cuda_error(
        cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, nullptr),
        "cudaMemcpyAsync H2D"
    );
}

auto copy_device_to_host(void* dst, const void* src, std::size_t bytes) -> void
{
    if (bytes == 0)
    {
        return;
    }

    throw_if_cuda_error(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
}

auto copy_device_to_host_async(void* dst, const void* src, std::size_t bytes) -> void
{
    if (bytes == 0)
    {
        return;
    }

    throw_if_cuda_error(
        cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, nullptr),
        "cudaMemcpyAsync D2H"
    );
}

auto copy_device_to_host_async_on_stream(
    void* dst,
    const void* src,
    std::size_t bytes,
    void* stream) -> void
{
    if (bytes == 0)
    {
        return;
    }

    throw_if_cuda_error(
        cudaMemcpyAsync(
            dst,
            src,
            bytes,
            cudaMemcpyDeviceToHost,
            static_cast<cudaStream_t>(stream)
        ),
        "cudaMemcpyAsync D2H"
    );
}

auto copy_device_to_device(void* dst, const void* src, std::size_t bytes) -> void
{
    if (bytes == 0)
    {
        return;
    }

    throw_if_cuda_error(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice), "cudaMemcpy D2D");
}

auto async_copy_stream_create() -> void*
{
    cudaStream_t stream{};
    throw_if_cuda_error(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    return static_cast<void*>(stream);
}

auto async_copy_stream_destroy(void* stream) noexcept -> void
{
    if (stream == nullptr)
    {
        return;
    }

    (void)cudaStreamDestroy(static_cast<cudaStream_t>(stream));
}

auto async_copy_stream_wait_for_fence(void* stream, void* fence) -> void
{
    throw_if_cuda_error(
        cudaStreamWaitEvent(
            static_cast<cudaStream_t>(stream),
            static_cast<cudaEvent_t>(fence),
            0
        ),
        "cudaStreamWaitEvent"
    );
}

auto async_copy_fence_create() -> void*
{
    cudaEvent_t event{};
    throw_if_cuda_error(
        cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
        "cudaEventCreateWithFlags"
    );
    return static_cast<void*>(event);
}

auto async_copy_fence_destroy(void* fence) noexcept -> void
{
    if (fence == nullptr)
    {
        return;
    }

    (void)cudaEventDestroy(static_cast<cudaEvent_t>(fence));
}

auto async_copy_fence_record(void* fence) -> void
{
    async_copy_fence_record_on_stream(fence, nullptr);
}

auto async_copy_fence_record_on_stream(void* fence, void* stream) -> void
{
    throw_if_cuda_error(
        cudaEventRecord(
            static_cast<cudaEvent_t>(fence),
            static_cast<cudaStream_t>(stream)
        ),
        "cudaEventRecord"
    );
}

auto async_copy_fence_wait(void* fence) -> void
{
    throw_if_cuda_error(
        cudaEventSynchronize(static_cast<cudaEvent_t>(fence)),
        "cudaEventSynchronize"
    );
}

auto profile_capture_start() -> void
{
    throw_if_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize before profiling");
    throw_if_cuda_error(cudaProfilerStart(), "cudaProfilerStart");
}

auto profile_capture_stop() -> void
{
    throw_if_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize after profiling");
    throw_if_cuda_error(cudaProfilerStop(), "cudaProfilerStop");
}

auto profile_range_push(const char* label) -> void
{
    nvtxRangePushA(label);
}

auto profile_range_pop() noexcept -> void
{
    (void)nvtxRangePop();
}

} // namespace amr::cuda
