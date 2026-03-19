#ifndef AMR_INCLUDED_CUDA_PERMUTATION
#define AMR_INCLUDED_CUDA_PERMUTATION

#include <cstddef>

namespace amr::cuda
{

auto permute_patches_inplace(
    void*              device_buffer,
    std::size_t        patch_count,
    std::size_t        patch_bytes,
    const std::size_t* host_sources,
    std::size_t        source_count
) -> void;

} // namespace amr::cuda

#endif // AMR_INCLUDED_CUDA_PERMUTATION
