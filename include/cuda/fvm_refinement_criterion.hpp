#ifndef AMR_INCLUDED_CUDA_FVM_REFINEMENT_CRITERION
#define AMR_INCLUDED_CUDA_FVM_REFINEMENT_CRITERION

#include <cstddef>
#include <cstdint>

namespace amr::cuda
{

struct scalar_patch_amr_launch_config
{
    std::size_t num_patches;
    std::size_t cells_per_patch;
    double      refine_threshold;
    double      coarsen_threshold;
    int         min_level;
    int         max_level;
};

auto compute_scalar_patch_amr_decisions(
    const double*                     packed_patch_data,
    std::size_t                       packed_value_count,
    const int*                        patch_levels,
    std::size_t                       level_count,
    const scalar_patch_amr_launch_config& config,
    std::int8_t*                      decisions,
    std::size_t                       decision_count
) -> void;

auto compute_scalar_patch_amr_decisions_from_device(
    const double*                        device_patch_data,
    const int*                           patch_levels,
    std::size_t                          level_count,
    const scalar_patch_amr_launch_config& config,
    std::int8_t*                         decisions,
    std::size_t                          decision_count
) -> void;

} // namespace amr::cuda

#endif // AMR_INCLUDED_CUDA_FVM_REFINEMENT_CRITERION
