#ifndef AMR_INCLUDED_CUDA_PROFILER
#define AMR_INCLUDED_CUDA_PROFILER

namespace amr::cuda
{

auto profile_capture_start() -> void;

auto profile_capture_stop() -> void;

} // namespace amr::cuda

#endif // AMR_INCLUDED_CUDA_PROFILER
