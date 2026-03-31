#include "cuda/fvm_time_step.hpp"
#include "solver/EulerPhysics.hpp"
#include "solver/AdvectionPhysics.hpp"
#include <cuda_runtime.h>
#include <array>
#include <cfloat>

namespace amr::cuda {

// Device helper to map an interior 1D index to a padded 1D index (skipping the halo)
template <int DIM>
__device__ __forceinline__ std::size_t get_padded_linear_idx(
    std::size_t interior_idx, 
    const time_step_launch_config& config)
{
    std::size_t padded_idx = 0;
    for (int d = 0; d < DIM; ++d) {
        // Extract the coordinate
        std::size_t coord = (interior_idx / config.data_strides[d]) % config.data_sizes[d];
        
        // Add the halo width to shift into the padded region
        padded_idx += (coord + config.halo_width) * config.padded_strides[d];
    }
    return padded_idx;
}

// Custom atomic minimum for double precision
__device__ __forceinline__ double atomicMinDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        if (__longlong_as_double(assumed) <= val) break;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val));
    } while (assumed != old);
    return __longlong_as_double(old);
}

template <typename EquationT, int DIM>
__device__ __forceinline__ void time_step_kernel_body(
    std::array<double*, EquationT::NVAR> in_patches, 
    std::array<double*, EquationT::NVAR> out_patches, 
    const int* patch_levels,
    time_step_launch_config config)
{
    // Flatten over the interior cells only
    const std::size_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    const std::size_t total_work_items = config.num_patches * config.data_flat_size;
    if (global_idx >= total_work_items) return;

    // Identify patch and local indices
    const std::size_t p_idx        = global_idx / config.data_flat_size;
    const std::size_t interior_idx = global_idx % config.data_flat_size;
    
    // Map to the actual memory layout (skipping the halo)
    const std::size_t local_padded_idx = get_padded_linear_idx<DIM>(interior_idx, config);
    const std::size_t memory_idx       = (p_idx * config.patch_flat_size) + local_padded_idx;

    // Compute Cell Size using the AMR Level
    int level = patch_levels[p_idx];
    double dt_over_dx[3];
    for (int d = 0; d < DIM; ++d) {
        // Bitshift (1 << level) divides the root size perfectly
        double c_size = config.root_c_size[d] / static_cast<double>(1 << level);
        dt_over_dx[d] = config.dt / c_size;
    }

    // Fetch Center Cell into Registers
    amr::containers::static_vector<double, EquationT::NVAR> U_center;
    for (int k = 0; k < EquationT::NVAR; ++k) {
        U_center[k] = in_patches[k][memory_idx];
    }

    amr::containers::static_vector<double, EquationT::NVAR> total_update{};

    // Compute Fluxes
    for (int d = 0; d < DIM; ++d) {
        const std::size_t stride = (d == 0) ? 1 : (d == 1 ? config.stride_y : config.stride_z);
        
        amr::containers::static_vector<double, EquationT::NVAR> U_L, U_R;
        for (int k = 0; k < EquationT::NVAR; ++k) {
            U_L[k] = in_patches[k][memory_idx - stride];
            U_R[k] = in_patches[k][memory_idx + stride];
        }

        amr::containers::static_vector<double, EquationT::NVAR> fL, fR;
        EquationT::rusanovFlux(U_L, U_center, fL, d, config.gamma);
        EquationT::rusanovFlux(U_center, U_R, fR, d, config.gamma);

        for (int k = 0; k < EquationT::NVAR; ++k) {
            total_update[k] -= dt_over_dx[d] * (fR[k] - fL[k]);
        }
    }

    // Write to Global Memory
    for (int k = 0; k < EquationT::NVAR; ++k) {
        out_patches[k][memory_idx] = U_center[k] + total_update[k];
    }
}

template <typename EquationT, int DIM>
__global__ void time_step_kernel_device_dt(
    std::array<double*, EquationT::NVAR> in_patches,
    std::array<double*, EquationT::NVAR> out_patches,
    const int* patch_levels,
    time_step_launch_config config,
    const double* global_dt)
{
    time_step_launch_config local_config = config;
    local_config.dt = (*global_dt) * config.cfl;
    time_step_kernel_body<EquationT, DIM>(
        in_patches,
        out_patches,
        patch_levels,
        local_config
    );
}

template <typename EquationT, int DIM>
auto launch_time_step_kernel_with_device_dt(
    std::array<double*, EquationT::NVAR> device_in_patches,
    std::array<double*, EquationT::NVAR> device_out_patches,
    const int* device_patch_levels,
    const time_step_launch_config& config,
    const double* device_dt_buffer) -> void
{
    if (config.num_patches == 0) return;

    const std::size_t total_work_items = config.num_patches * config.data_flat_size;

    constexpr unsigned int threads_per_block = 256;
    const unsigned int blocks = (total_work_items + threads_per_block - 1) / threads_per_block;

    time_step_kernel_device_dt<EquationT, DIM><<<blocks, threads_per_block>>>(
        device_in_patches,
        device_out_patches,
        device_patch_levels,
        config,
        device_dt_buffer
    );

    cudaGetLastError();
}

template <typename EquationT, int DIM>
__global__ void compute_dt_kernel(
    std::array<const double*, EquationT::NVAR> in_patches, 
    const int* patch_levels,
    time_step_launch_config config,
    double* global_dt)
{
    const std::size_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t total_work_items = config.num_patches * config.data_flat_size;
    
    double local_dt = DBL_MAX;

    if (global_idx < total_work_items) {
        const std::size_t p_idx        = global_idx / config.data_flat_size;
        const std::size_t interior_idx = global_idx % config.data_flat_size;
        const std::size_t local_padded_idx = get_padded_linear_idx<DIM>(interior_idx, config);
        const std::size_t memory_idx       = (p_idx * config.patch_flat_size) + local_padded_idx;

        int level = patch_levels[p_idx];

        for (int d = 0; d < DIM; ++d) {
            // Re-using the memory_idx to fetch directly from the SoA arrays
            double speed = EquationT::getMaxSpeed(in_patches, memory_idx, d, config.gamma);
            if (speed > 1e-12) {
                double c_size = config.root_c_size[d] / static_cast<double>(1 << level);
                local_dt = min(local_dt, c_size / speed);
            }
        }
    }

    // Warp-level reduction (32 threads collaborate to find the minimum)
    unsigned int mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2) {
        local_dt = min(local_dt, __shfl_down_sync(mask, local_dt, offset));
    }

    // Only the first thread in each warp writes to global memory
    if (threadIdx.x % 32 == 0 && local_dt < DBL_MAX) {
        atomicMinDouble(global_dt, local_dt);
    }
}

__global__ void initialize_double_buffer_kernel(double* buffer, double value)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        *buffer = value;
    }
}

__global__ void accumulate_scaled_double_buffer_kernel(
    double*       accumulator,
    const double* value,
    double        scale)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        *accumulator += (*value) * scale;
    }
}

auto launch_set_double_buffer(double* device_buffer, double value) -> void
{
    initialize_double_buffer_kernel<<<1, 1>>>(device_buffer, value);
    cudaGetLastError();
}

auto launch_accumulate_scaled_double_buffer(
    double* device_accumulator,
    const double* device_value,
    double scale) -> void
{
    accumulate_scaled_double_buffer_kernel<<<1, 1>>>(
        device_accumulator,
        device_value,
        scale
    );
    cudaGetLastError();
}

template <typename EquationT, int DIM>
auto launch_compute_dt_kernel_device(
    std::array<const double*, EquationT::NVAR> device_in_patches, 
    const int* device_patch_levels,
    const time_step_launch_config& config,
    double* device_dt_buffer) -> void
{
    if (config.num_patches == 0) return;

    launch_set_double_buffer(device_dt_buffer, DBL_MAX);

    const std::size_t total_work_items = config.num_patches * config.data_flat_size;
    constexpr unsigned int threads_per_block = 256;
    const unsigned int blocks = (total_work_items + threads_per_block - 1) / threads_per_block;

    compute_dt_kernel<EquationT, DIM><<<blocks, threads_per_block>>>(
        device_in_patches, device_patch_levels, config, device_dt_buffer
    );
    cudaGetLastError();
}

// Explicit instantiation for the exact signatures
template auto launch_time_step_kernel_with_device_dt<EulerPhysics<2>, 2>(
    std::array<double*, 4>,
    std::array<double*, 4>,
    const int*,
    const time_step_launch_config&,
    const double*
) -> void;

template auto launch_time_step_kernel_with_device_dt<EulerPhysics<3>, 3>(
    std::array<double*, 5>,
    std::array<double*, 5>,
    const int*,
    const time_step_launch_config&,
    const double*
) -> void;

template auto launch_time_step_kernel_with_device_dt<AdvectionPhysics<2>, 2>(
    std::array<double*, 1>,
    std::array<double*, 1>,
    const int*,
    const time_step_launch_config&,
    const double*
) -> void;

template auto launch_time_step_kernel_with_device_dt<AdvectionPhysics<3>, 3>(
    std::array<double*, 1>,
    std::array<double*, 1>,
    const int*,
    const time_step_launch_config&,
    const double*
) -> void;

template auto launch_compute_dt_kernel_device<EulerPhysics<2>, 2>(
    std::array<const double*, 4>, const int*, const time_step_launch_config&, double*
) -> void;

template auto launch_compute_dt_kernel_device<EulerPhysics<3>, 3>(
    std::array<const double*, 5>, const int*, const time_step_launch_config&, double*
) -> void;

template auto launch_compute_dt_kernel_device<AdvectionPhysics<2>, 2>(
    std::array<const double*, 1>, const int*, const time_step_launch_config&, double*
) -> void;

template auto launch_compute_dt_kernel_device<AdvectionPhysics<3>, 3>(
    std::array<const double*, 1>, const int*, const time_step_launch_config&, double*
) -> void;

} // namespace amr::cuda
