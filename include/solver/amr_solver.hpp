#ifndef AMR_SOLVER_HPP
#define AMR_SOLVER_HPP

#include "ndtree/ndtree.hpp"
#include "cell_types.hpp"
#include "EulerPhysics.hpp"
#include "physics_system.hpp"
#include <vector>
#include <functional>

/**
 * @brief AMR Solver
 * @tparam TreeT The ndtree structure
 * @tparam GeometryT The physics_system (coordinates/sizes)
 * @tparam EquationT The physical model (Euler, Advection, etc.)
 * @tparam DIM Dimensionality
 */
template<typename TreeT, typename GeometryT, typename EquationT, int DIM>
class amr_solver {
private:
    TreeT m_tree;
    double gamma;       // Specific heat ratio
    double cfl;         // CFL number

    // Flat buffers for the "New" state on the GPU
    double* d_rho_dest;
    double* d_rhou_dest;
    double* d_rhov_dest;
    double* d_energy_dest;

    // Table to tell the kernel where to read and where to write
    GpuPatchTable* d_patch_table;

public:
    using PatchLayoutT = typename TreeT::patch_layout_t;
    using PatchIndexT = typename TreeT::patch_index_t;
    static constexpr int NVAR = EquationT::NVAR;

    amr_solver(size_t capacity, double gamma_ = 1.4, double cfl_ = 0.1)
        : m_tree(capacity), gamma(gamma_), cfl(cfl_) {
        // Dummy dimensions were removed from new EulerPhysics.hpp
        static_assert(DIM == 2 || DIM == 3, "Error: Wrong dimensions");

        size_t total_cells = capacity * PatchLayoutT::flat_size();  // TODO: too huge?
        
        // Allocate Destination Buffers on GPU
        cudaMalloc(&d_rho_dest,    total_cells * sizeof(double));
        cudaMalloc(&d_rhou_dest,   total_cells * sizeof(double));
        cudaMalloc(&d_rhov_dest,   total_cells * sizeof(double));
        cudaMalloc(&d_energy_dest, total_cells * sizeof(double));
        
        // Allocate the Table on GPU
        cudaMalloc(&d_patch_table, capacity * sizeof(GpuPatchTable));
    }

    ~amr_solver() {
        cudaFree(d_rho_dest);
        cudaFree(d_rhou_dest);
        cudaFree(d_rhov_dest);
        cudaFree(d_energy_dest);
        cudaFree(d_patch_table);
    }

    TreeT& get_tree() {
        return m_tree;
    }

    /**
     * @brief Helper to gather the full conservative state from a specific cell
     */
    auto get_full_state(std::size_t patch_idx, std::size_t linear_idx) const {
        auto patch_id = m_tree.get_node_index_at(patch_idx);
        amr::containers::static_vector<double, NVAR> state;

        // Static loop over the FieldTags defined in the Equation policy
        auto fill_state = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            ((state[Is] = m_tree.template get_patch<
                typename std::tuple_element<Is, typename EquationT::FieldTags>::type
              >(patch_id)[linear_idx]), ...);
        };
        fill_state(std::make_index_sequence<NVAR>{});
        
        return state;
    }

    void set_full_state(std::size_t patch_idx, std::size_t linear_idx, 
                        const amr::containers::static_vector<double, NVAR>& state) {
        auto patch_id = m_tree.get_node_index_at(patch_idx);
        
        auto write_state = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            ((m_tree.template get_patch<
                typename std::tuple_element<Is, typename EquationT::FieldTags>::type
              >(patch_id)[linear_idx] = state[Is]), ...);
        };
        write_state(std::make_index_sequence<NVAR>{});
    }

    double get_gamma() const {
        return gamma;
    }

    double get_cfl() const {
        return cfl;
    }

    template<typename InitFunc>
    void initialize(InitFunc init_func) {
        for (std::size_t p_idx = 0; p_idx < m_tree.size(); ++p_idx) {
            auto patch_id = m_tree.get_node_index_at(p_idx);
            auto c_size   = GeometryT::cell_sizes(patch_id);

            for (std::size_t l_idx = 0; l_idx < PatchLayoutT::flat_size(); ++l_idx) {
                if (amr::ndt::utils::patches::is_halo_cell<PatchLayoutT>(l_idx)) continue;

                // Dimension-agnostic coordinate fetch
                auto cell_origin = GeometryT::cell_coord(patch_id, l_idx);
                std::array<double, DIM> coords;
                for(int d=0; d<DIM; ++d) coords[d] = cell_origin[d] + 0.5 * c_size[d];

                // IC -> Primitive -> Conservative
                auto prim = init_func(coords);
                amr::containers::static_vector<double, NVAR> cons;
                EquationT::primitiveToConservative(prim, cons, gamma);
                
                set_full_state(p_idx, l_idx, cons);
            }
        }
    }

    void time_step(double dt) {
        constexpr size_t patch_flat_size = PatchLayoutT::flat_size();
        // Stride for moving "up" or "down" in the patch (Y-direction)
        constexpr size_t stride_y = PatchLayoutT::padded_layout_t::shape_t::sizes()[DIM-1];
        // Stride for Z (only used if DIM=3)
        constexpr size_t stride_z = (DIM == 3) ? 
            PatchLayoutT::padded_layout_t::shape_t::sizes()[1] * stride_y : 0;

        for (std::size_t p_idx = 0; p_idx < m_tree.size(); ++p_idx) {
            auto patch_id = m_tree.get_node_index_at(p_idx);
            auto c_size   = GeometryT::cell_sizes(patch_id);

            std::vector<amr::containers::static_vector<double, NVAR>> update_buffer(patch_flat_size);

            for (std::size_t l_idx = 0; l_idx < patch_flat_size; ++l_idx) {
                if (amr::ndt::utils::patches::is_halo_cell<PatchLayoutT>(l_idx)) continue;

                auto U_cell = get_full_state(p_idx, l_idx);
                amr::containers::static_vector<double, NVAR> total_update = amr::containers::static_vector<double, NVAR>{};

                // Generic loop over dimensions (X, Y, Z)
                for (int d = 0; d < DIM; ++d) {
                    size_t stride = (d == 0) ? 1 : (d == 1 ? stride_y : stride_z);
                    
                    auto U_L = get_full_state(p_idx, l_idx - stride);
                    auto U_R = get_full_state(p_idx, l_idx + stride);

                    amr::containers::static_vector<double, NVAR> fL, fR;
                    EquationT::rusanovFlux(U_L, U_cell, fL, d, gamma);
                    EquationT::rusanovFlux(U_cell, U_R, fR, d, gamma);

                    for (int k = 0; k < NVAR; ++k) {
                        total_update[k] -= (dt / c_size[d]) * (fR[k] - fL[k]);
                    }
                }

                for (int k = 0; k < NVAR; ++k) {
                    update_buffer[l_idx][k] = U_cell[k] + total_update[k];
                }
            }

            for (std::size_t l_idx = 0; l_idx < patch_flat_size; ++l_idx) {
                if (!amr::ndt::utils::patches::is_halo_cell<PatchLayoutT>(l_idx)) {
                    set_full_state(p_idx, l_idx, update_buffer[l_idx]);
                }
            }
        }
    }

    void gpu_time_step(double dt) {
        size_t num_patches = m_tree.size();
        size_t cells_per_patch = PatchLayoutT::flat_size();
        if (num_patches == 0) return;

        auto first_patch_id = m_tree.get_node_index_at(0);
        auto c_size = GeometryT::cell_sizes(first_patch_id);
        int stride_y = PatchLayoutT::padded_layout_t::shape_t::sizes()[1];

        // Build the Pointer Table on Host
        std::vector<GpuPatchTable> host_table(num_patches);
        for (size_t p = 0; p < num_patches; ++p) {
            auto id = m_tree.get_node_index_at(p);
            
            // SOURCE: Direct pointers into the ndtree's existing contiguous arrays
            host_table[p].fields_src[0] = m_tree.template get_patch<amr::cell::Rho>(id).data();
            host_table[p].fields_src[1] = m_tree.template get_patch<amr::cell::Rhou>(id).data();
            host_table[p].fields_src[2] = m_tree.template get_patch<amr::cell::Rhov>(id).data();
            host_table[p].fields_src[3] = m_tree.template get_patch<amr::cell::E2D>(id).data();

            // DESTINATION: Offsets into your separate flat GPU destination buffers
            host_table[p].fields_dst[0] = d_rho_dest + (p * cells_per_patch);
            host_table[p].fields_dst[1] = d_rhou_dest + (p * cells_per_patch);
            host_table[p].fields_dst[2] = d_rhov_dest + (p * cells_per_patch);
            host_table[p].fields_dst[3] = d_energy_dest + (p * cells_per_patch);
        }

        // Upload Table to Device
        cudaMemcpy(d_patch_table, host_table.data(), 
                num_patches * sizeof(GpuPatchTable), 
                cudaMemcpyHostToDevice);

        // Launch Kernel
        // Grid size = number of patches, Block size = cells per patch
        // Each block handles one Patch, each thread handles one Cell
        euler_2d_kernel<<<num_patches, cells_per_patch>>>(
            d_patch_table, dt, c_size[0], c_size[1], gamma, stride_y
        );
        cudaDeviceSynchronize();

        // SYNC BACK: Copy the flat destination buffers back into the ndtree
        for (size_t p = 0; p < num_patches; ++p) {
            auto id = m_tree.get_node_index_at(p);
            cudaMemcpy(m_tree.template get_patch<amr::cell::Rho>(id).data(), 
                    d_rho_dest + (p * cells_per_patch), 
                    cells_per_patch * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(m_tree.template get_patch<amr::cell::Rhou>(id).data(), 
                    d_rhou_dest + (p * cells_per_patch), 
                    cells_per_patch * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(m_tree.template get_patch<amr::cell::Rhov>(id).data(), 
                    d_rhov_dest + (p * cells_per_patch), 
                    cells_per_patch * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(m_tree.template get_patch<amr::cell::E2D>(id).data(), 
                    d_energy_dest + (p * cells_per_patch), 
                    cells_per_patch * sizeof(double), cudaMemcpyDeviceToHost);
        }
    }

    double compute_time_step() const {
        double dt_min = std::numeric_limits<double>::max();

        for (std::size_t p_idx = 0; p_idx < m_tree.size(); ++p_idx) {
            auto patch_id = m_tree.get_node_index_at(p_idx);
            auto c_size = GeometryT::cell_sizes(patch_id);

            for (std::size_t l_idx = 0; l_idx < PatchLayoutT::flat_size(); ++l_idx) {
                if (amr::ndt::utils::patches::is_halo_cell<PatchLayoutT>(l_idx)) continue;

                auto U = get_full_state(p_idx, l_idx);
                
                // Ask Equation for max wave speed in each direction
                for (int d = 0; d < DIM; ++d) {
                    double speed = EquationT::getMaxSpeed(U, d, gamma);
                    if (speed > 1e-12) {
                        dt_min = std::min(dt_min, c_size[d] / speed);
                    }
                }
            }
        }
        return cfl * dt_min;
    }
};

// Typedefs
template<typename TreeT, typename GeometryT, typename EquationT>
using amr_solver_2d = amr_solver<TreeT, GeometryT, EquationT, 2>;

template<typename TreeT, typename GeometryT, typename EquationT>
using amr_solver_3d = amr_solver<TreeT, GeometryT, EquationT, 3>;

// GPU Kernel for 2D Euler
__global__ void euler_2d_kernel(
        GpuPatchTable* patches, 
        double dt, double dx, double dy, double gamma, 
        int stride_y) 
{
    int p_idx = blockIdx.x;  // One block per patch
    int l_idx = threadIdx.x;  // One thread per cell (assuming patch is small enough)

    // Load data for this cell (U_cell)
    double U_cell[4];
    for(int k=0; k<4; ++k) U_cell[k] = patches[p_idx].fields_src[k][l_idx];

    // Halo Check: Skip calculations for ghost cells
    // TODO: how to access? needs to become a device function
    // If is_halo_cell calls other functions or uses class methods, all of those must also be marked __host__ __device__
    if (amr::ndt::utils::patches::is_halo_cell<PatchLayoutT>(l_idx)) return; 

    double total_update[4] = {0.0, 0.0, 0.0, 0.0};

    // X-Direction Fluxes (Stride = 1)
    {
        double U_L[4], U_R[4], fL[4], fR[4];
        for(int k=0; k<4; ++k) {
            U_L[k] = patches[p_idx].fields_src[k][l_idx - 1];
            U_R[k] = patches[p_idx].fields_src[k][l_idx + 1];
        }

        // TODO: how to access? needs to become a device function
        // If rusanovFlux calls other functions or uses class methods, all of those must also be marked __host__ __device__
        EulerPhysics2D::rusanovFlux(U_L, U_cell, fL, 0, gamma); // direction 0 = X
        EulerPhysics2D::rusanovFlux(U_cell, U_R, fR, 0, gamma);

        for(int k=0; k<4; ++k) total_update[k] -= (dt / dx) * (fR[k] - fL[k]);
    }

    // Y-Direction Fluxes (Stride = stride_y)
    {
        double U_L[4], U_R[4], fL[4], fR[4];
        for(int k=0; k<4; ++k) {
            U_L[k] = patches[p_idx].fields_src[k][l_idx - stride_y];
            U_R[k] = patches[p_idx].fields_src[k][l_idx + stride_y];
        }

        EulerPhysics2D::rusanovFlux(U_L, U_cell, fL, 1, gamma); // direction 1 = Y
        EulerPhysics2D::rusanovFlux(U_cell, U_R, fR, 1, gamma);

        for(int k=0; k<4; ++k) total_update[k] -= (dt / dy) * (fR[k] - fL[k]);
    }

    // 5. Write back to DESTINATION patches to avoid race conditions
    for(int k=0; k<4; ++k) {
        patches[p_idx].fields_dst[k][l_idx] = U_cell[k] + total_update[k];
    }
}

#endif // AMR_SOLVER_HPP