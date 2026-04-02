#ifndef AMR_INCLUDED_CUDA_FVM_TIME_STEP
#define AMR_INCLUDED_CUDA_FVM_TIME_STEP

#include <cstddef>
#include <cstdint>
#include <array>

namespace amr::cuda
{
    struct time_step_launch_config
    {
        std::size_t num_patches;
        std::size_t patch_flat_size;
        std::size_t data_flat_size; // The number of interior cells
        std::size_t halo_width;  
        
        double dt;
        double cfl;
        double gamma;
        std::array<double, 3> root_c_size;

        std::size_t stride_y;
        std::size_t stride_z;
        
        std::array<std::size_t, 3> data_sizes;
        std::array<std::size_t, 3> data_strides;
        std::array<std::size_t, 3> padded_strides;
    };

    template <typename EquationT, int DIM>
    auto launch_time_step_kernel_with_device_dt(
        std::array<double*, EquationT::NVAR> device_in_patches,
        std::array<double*, EquationT::NVAR> device_out_patches,
        const int* device_patch_levels,
        const time_step_launch_config& config,
        const double* device_dt_buffer
    ) -> void;

    template <typename EquationT, int DIM>
    auto launch_compute_dt_kernel_device(
        std::array<const double*, EquationT::NVAR> device_in_patches,
        const int* device_patch_levels,
        const time_step_launch_config& config,
        double* device_dt_buffer
    ) -> void;

    auto launch_set_double_buffer(double* device_buffer, double value) -> void;

    auto launch_set_uint32_buffer(std::uint32_t* device_buffer, std::uint32_t value)
        -> void;

    auto launch_finalize_step_dt(
        double*         device_dt_buffer,
        double*         device_dt_accumulator,
        double*         device_remaining_time,
        std::uint32_t*  device_executed_step_count,
        double          cfl
    ) -> void;

} // namespace amr::cuda

#endif // AMR_INCLUDED_CUDA_FVM_TIME_STEP
