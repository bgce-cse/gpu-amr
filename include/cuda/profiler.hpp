#ifndef AMR_INCLUDED_CUDA_PROFILER
#define AMR_INCLUDED_CUDA_PROFILER

namespace amr::cuda
{

auto profile_capture_start() -> void;

auto profile_capture_stop() -> void;

auto profile_range_push(const char* label) -> void;

auto profile_range_pop() noexcept -> void;

class scoped_profile_range
{
public:
    explicit scoped_profile_range(const char* label)
    {
        profile_range_push(label);
    }

    ~scoped_profile_range()
    {
        profile_range_pop();
    }

    scoped_profile_range(scoped_profile_range const&) = delete;
    auto operator=(scoped_profile_range const&) -> scoped_profile_range& = delete;
};

} // namespace amr::cuda

#endif // AMR_INCLUDED_CUDA_PROFILER
