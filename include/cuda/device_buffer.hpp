#ifndef AMR_INCLUDED_CUDA_DEVICE_BUFFER
#define AMR_INCLUDED_CUDA_DEVICE_BUFFER

#include <cstddef>

namespace amr::cuda
{

auto device_malloc(std::size_t bytes) -> void*;

auto device_free(void* ptr) noexcept -> void;

auto host_pinned_malloc(std::size_t bytes) -> void*;

auto host_pinned_free(void* ptr) noexcept -> void;

auto copy_host_to_device(void* dst, const void* src, std::size_t bytes) -> void;

auto copy_host_to_device_async(void* dst, const void* src, std::size_t bytes) -> void;

auto copy_device_to_host(void* dst, const void* src, std::size_t bytes) -> void;

auto copy_device_to_host_async(void* dst, const void* src, std::size_t bytes) -> void;

auto copy_device_to_host_async_on_stream(
    void* dst,
    const void* src,
    std::size_t bytes,
    void* stream
) -> void;

auto copy_device_to_device(void* dst, const void* src, std::size_t bytes) -> void;

auto async_copy_stream_create() -> void*;

auto async_copy_stream_destroy(void* stream) noexcept -> void;

auto async_copy_stream_wait_for_fence(void* stream, void* fence) -> void;

auto async_copy_fence_create() -> void*;

auto async_copy_fence_destroy(void* fence) noexcept -> void;

auto async_copy_fence_record(void* fence) -> void;

auto async_copy_fence_record_on_stream(void* fence, void* stream) -> void;

auto async_copy_fence_wait(void* fence) -> void;

} // namespace amr::cuda

#endif // AMR_INCLUDED_CUDA_DEVICE_BUFFER
