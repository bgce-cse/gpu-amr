#ifndef AMR_INCLUDED_CUDA_PERMUTATION
#define AMR_INCLUDED_CUDA_PERMUTATION

#include <cstddef>
#include <span>

namespace amr::cuda
{

auto permute_patches_inplace_batch(
    std::span<void**>            device_buffer_slots,
    std::span<const std::size_t> patch_bytes,
    std::size_t                  patch_count,
    const std::size_t*           host_sources,
    std::size_t                  source_count
) -> void;

} // namespace amr::cuda

#endif // AMR_INCLUDED_CUDA_PERMUTATION
