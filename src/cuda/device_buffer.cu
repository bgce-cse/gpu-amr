#include "cuda/device_buffer.hpp"

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

auto copy_host_to_device(void* dst, const void* src, std::size_t bytes) -> void
{
    if (bytes == 0)
    {
        return;
    }

    throw_if_cuda_error(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D");
}

auto copy_device_to_host(void* dst, const void* src, std::size_t bytes) -> void
{
    if (bytes == 0)
    {
        return;
    }

    throw_if_cuda_error(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
}

auto copy_device_to_device(void* dst, const void* src, std::size_t bytes) -> void
{
    if (bytes == 0)
    {
        return;
    }

    throw_if_cuda_error(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice), "cudaMemcpy D2D");
}

} // namespace amr::cuda
