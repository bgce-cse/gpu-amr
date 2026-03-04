#ifndef AMR_SOLVER_MUSCL_HPP
#define AMR_SOLVER_MUSCL_HPP

#include "ndtree/ndtree.hpp"
#include "cell_types.hpp"
#include "EulerPhysics.hpp"
#include "physics_system.hpp"
#include <vector>
#include <functional>

template<typename TreeT, typename PhysicsT, int DIM>
class amr_solver_MUSCL {
private:
    TreeT m_tree;
    double gamma;       // Specific heat ratio
    double cfl;         // CFL number

public:
    using physics_system_t = PhysicsT;
    using PatchLayoutT = typename TreeT::patch_layout_t;
    using PatchIndexT = typename TreeT::patch_index_t;
    using EulerPhysicsSolver = EulerPhysics<DIM>;
    static constexpr int NVAR = EulerPhysicsSolver::NVAR;

    amr_solver_MUSCL(size_t capacity, double gamma_ = 1.4, double cfl_ = 0.3)  //0.3
        : m_tree(capacity), gamma(gamma_), cfl(cfl_) {
        // Dummy dimensions were removed from new EulerPhysics.hpp
        static_assert(DIM == 2 || DIM == 3, "Error: Wrong dimensions");
        }

    TreeT& get_tree() {
        return m_tree;
    }

    double get_gamma() const {
        return gamma;
    }

    double get_cfl() const {
        return cfl;
    }

    template<typename InitFunc>
    void initialize(InitFunc init_func) {
        if constexpr (DIM == 2) {
            initialize_2d(init_func);
        } else if constexpr (DIM == 3) {
            initialize_3d(init_func);
        }
    }

    void time_step(double dt) {
        if constexpr (DIM == 2) {
            time_step_2d(dt);
        } else if constexpr (DIM == 3) {
            time_step_3d(dt);
        }
    }

    double compute_time_step() const {
        if constexpr (DIM == 2) {
            return compute_time_step_2d();
        } else if constexpr (DIM == 3) {
            return compute_time_step_3d();
        }
        return 0.0;
    }

    template<typename InitFunc>
    void initialize_2d(InitFunc init_func) {
        constexpr std::size_t patch_flat_size = PatchLayoutT::flat_size();

        // Loop over all patches
        for (std::size_t patch_idx = 0; patch_idx < m_tree.size(); ++patch_idx) {
            
            auto patch_id = m_tree.get_node_index_at(patch_idx);
            
            // Get references to the patch data buffers
            auto& rho_patch  = m_tree.template get_patch<amr::cell::Rho>(patch_idx);
            auto& rhou_patch = m_tree.template get_patch<amr::cell::Rhou>(patch_idx);
            auto& rhov_patch = m_tree.template get_patch<amr::cell::Rhov>(patch_idx);
            auto& e_patch    = m_tree.template get_patch<amr::cell::E2D>(patch_idx);

            // Loop over cells in the patch and set IC
            for (std::size_t linear_idx = 0; linear_idx != patch_flat_size; ++linear_idx) {
                
                // Skip Halo Cells on initialization
                if (amr::ndt::utils::patches::is_halo_cell<PatchLayoutT>(linear_idx)) {
                    continue;
                }

                // Get cell center coordinates using physics system
                auto cell_center = physics_system_t::cell_coord(patch_id, linear_idx);
                
                // For cell centers, add 0.5 * cell_size
                auto cell_size = physics_system_t::cell_sizes(patch_id);
                double x_world = cell_center[0] + 0.5 * cell_size[0];
                double y_world = cell_center[1] + 0.5 * cell_size[1];

                // Evaluate the Initial Condition
                amr::containers::static_vector<double, NVAR> prim = init_func(x_world, y_world);

                // Convert to conservative and write to patch
                amr::containers::static_vector<double, NVAR> cons;
                EulerPhysicsSolver::primitiveToConservative(prim, cons, gamma);
                
                rho_patch[linear_idx]  = cons[0];
                rhou_patch[linear_idx] = cons[1];
                rhov_patch[linear_idx] = cons[2];
                e_patch[linear_idx]    = cons[3];
            }
        }
    }

    template<typename InitFunc>
    void initialize_3d(InitFunc init_func) {
        // Get geometrical constants
        // constexpr double global_size = 1.0;
        //constexpr size_t patch_size_x = Patch_Layout::data_layout_t::sizes()[0]; 
        constexpr std::size_t patch_size_padded_x = PatchLayoutT::padded_layout_t::shape_t::sizes()[2];
        constexpr std::size_t patch_size_padded_y = PatchLayoutT::padded_layout_t::shape_t::sizes()[1];
        constexpr std::size_t patch_size_padded_z = PatchLayoutT::padded_layout_t::shape_t::sizes()[0];
        constexpr std::size_t patch_flat_size = PatchLayoutT::flat_size();

        // Get max_depth from the TreeT definition
        constexpr auto max_depth = PatchIndexT::max_depth();

        // The total coordinate range is 2^max_depth (e.g., 512 for max_depth=9)
        constexpr double max_cell_size = 1u << max_depth;
        constexpr double max_coord_x = max_cell_size * patch_size_padded_x;
        constexpr double max_coord_y = max_cell_size * patch_size_padded_y;
        constexpr double max_coord_z = max_cell_size * patch_size_padded_z;

        // Loop over all patches
        for (std::size_t patch_idx = 0; patch_idx < m_tree.size(); ++patch_idx) {
            
            auto patch_id = m_tree.get_node_index_at(patch_idx);
            auto level = patch_id.level();
            auto [patch_coords_raw, _] = PatchIndexT::decode(patch_id.id());
            
            // Size of a single cell in terms of the maximum coordinate units
            uint32_t cell_size_units = 1u << (max_depth - level);
            
            // Get references to the patch data buffers
            auto& rho_patch  = m_tree.template get_patch<amr::cell::Rho>(patch_idx);
            auto& rhou_patch = m_tree.template get_patch<amr::cell::Rhou>(patch_idx);
            auto& rhov_patch = m_tree.template get_patch<amr::cell::Rhov>(patch_idx);
            auto& rhow_patch = m_tree.template get_patch<amr::cell::Rhow>(patch_idx);
            auto& e_patch    = m_tree.template get_patch<amr::cell::E3D>(patch_idx);

            // Loop over cells in the patch and set IC
            for (std::size_t linear_idx = 0; linear_idx != patch_flat_size; ++linear_idx) {
                
                // Skip Halo Cells on initialization
                if (amr::ndt::utils::patches::is_halo_cell<PatchLayoutT>(linear_idx)) {
                    continue;
                }

                // Calculate 3D coordinates (i, j, k) within the **unpadded** patch layout
                uint32_t i = static_cast<uint32_t>(linear_idx % patch_size_padded_x);
                uint32_t j = static_cast<uint32_t>((linear_idx / patch_size_padded_x) % patch_size_padded_y);
                uint32_t k = static_cast<uint32_t>(linear_idx / (patch_size_padded_x * patch_size_padded_y));
                
                // Calculate absolute cell center coordinates in max_coord units
                // x_units = (Patch Origin X) + (Cell Index i) * (Cell Size in units) + (Half Cell Size)
                uint32_t cell_x_units = patch_coords_raw[2] * cell_size_units + 
                                        static_cast<uint32_t>(i) * cell_size_units + 
                                        cell_size_units / 2;
                
                uint32_t cell_y_units = patch_coords_raw[1] * cell_size_units + 
                                        static_cast<uint32_t>(j) * cell_size_units + 
                                        cell_size_units / 2;
                
                uint32_t cell_z_units = patch_coords_raw[0] * cell_size_units + 
                                        static_cast<uint32_t>(k) * cell_size_units + 
                                        cell_size_units / 2;

                // Convert to normalized world coordinates [0, 1]
                double x_world = static_cast<double>(cell_x_units) / max_coord_x;
                double y_world = static_cast<double>(cell_y_units) / max_coord_y;
                double z_world = static_cast<double>(cell_z_units) / max_coord_z;


                // Evaluate the Initial Condition
                amr::containers::static_vector<double, NVAR> prim = init_func(x_world, y_world, z_world);

                // Convert to conservative and write to patch
                amr::containers::static_vector<double, NVAR> cons;
                EulerPhysicsSolver::primitiveToConservative(prim, cons, gamma);
                
                rho_patch[linear_idx]  = cons[0];
                rhou_patch[linear_idx] = cons[1];
                rhov_patch[linear_idx] = cons[2];
                rhow_patch[linear_idx] = cons[3];
                e_patch[linear_idx]    = cons[4];
            }
        }
    }

    void time_step_2d(double dt){
        constexpr size_t patch_size_padded_x = PatchLayoutT::padded_layout_t::shape_t::sizes()[1];
        constexpr size_t patch_flat_size = PatchLayoutT::flat_size();

        // Loop over the tree and apply the finite volume update
        for (std::size_t patch_idx = 0; patch_idx < m_tree.size(); ++patch_idx) {
            
            auto patch_id = m_tree.get_node_index_at(patch_idx);
            
            // Get cell sizes for this patch using physics system
            auto cell_size = physics_system_t::cell_sizes(patch_id);
            double dx = cell_size[0];
            double dy = cell_size[1];
            
            // Get current patch's conservative states
            auto& rho_patch = m_tree.template get_patch<amr::cell::Rho>(patch_idx);
            auto& rhou_patch = m_tree.template get_patch<amr::cell::Rhou>(patch_idx);
            auto& rhov_patch = m_tree.template get_patch<amr::cell::Rhov>(patch_idx);
            auto& e_patch = m_tree.template get_patch<amr::cell::E2D>(patch_idx);

            // temporary buffer for the new state of this patch
            std::vector<std::array<double, NVAR>> U_new_patch_buffer(patch_flat_size);

            // Loop over cells of patch
            for (std::size_t linear_idx = 0; linear_idx != patch_flat_size; ++linear_idx)
            {
                // Skip halo cells
                if (amr::ndt::utils::patches::is_halo_cell<PatchLayoutT>(linear_idx))
                {
                    continue;
                }

                amr::containers::static_vector<double, NVAR> UL, UR;

                // Extract the current conservative state U_cell
                amr::containers::static_vector<double, NVAR> U_cell = {
                    rho_patch[linear_idx],
                    rhou_patch[linear_idx],
                    rhov_patch[linear_idx],
                    e_patch[linear_idx]
                };

                // Placeholder for fluxes on each face
                amr::containers::static_vector<double, NVAR> Fx_minus;
                amr::containers::static_vector<double, NVAR> Fx_plus;
                amr::containers::static_vector<double, NVAR> Fy_plus;
                amr::containers::static_vector<double, NVAR> Fy_minus;

                // --- Calculate fluxes for the X-direction ---
                amr::containers::static_vector<double, NVAR> U_xp1 = {
                    rho_patch[linear_idx + 1],
                    rhou_patch[linear_idx + 1],
                    rhov_patch[linear_idx + 1],
                    e_patch[linear_idx + 1]
                };
                
                amr::containers::static_vector<double, NVAR> U_xm1 = {
                    rho_patch[linear_idx - 1],
                    rhou_patch[linear_idx - 1],
                    rhov_patch[linear_idx - 1],
                    e_patch[linear_idx - 1]
                };

                amr::containers::static_vector<double, NVAR> U_xm2 = {
                    rho_patch[linear_idx - 2],
                    rhou_patch[linear_idx - 2],
                    rhov_patch[linear_idx - 2],
                    e_patch[linear_idx - 2]
                };

                // Reconstruction with higher-order scheme (MUSCL)
                reconstructMUSCL(
                    U_xm1,   // i-1
                    U_cell,       // i
                    U_xp1,  // i+1
                    UL,       // U_{i+1/2}^-  (from cell i)
                    UR        // U_{i+1/2}^+  (from cell i+1)
                );

                EulerPhysicsSolver::rusanovFlux(UL, UR, Fx_plus, Direction::X, gamma);   // Use (i−1, i, i+1) to build states at i+1/2

                // Reconstruction with higher-order scheme (MUSCL)
                reconstructMUSCL(
                    U_xm2,   // i-2
                    U_xm1,    // i-1
                    U_cell,  // i
                    UL,       // U_{i-1/2}^-  (from cell i-1)
                    UR        // U_{i-1/2}^+  (from cell i)
                );

                EulerPhysicsSolver::rusanovFlux(UL, UR, Fx_minus, Direction::X, gamma);   // Use (i−2, i-1, i) to build states at i-1/2

                // --- Calculate fluxes for the Y-direction ---
                amr::containers::static_vector<double, NVAR> U_ym1 = {
                    rho_patch[linear_idx - patch_size_padded_x],
                    rhou_patch[linear_idx - patch_size_padded_x],
                    rhov_patch[linear_idx - patch_size_padded_x],
                    e_patch[linear_idx - patch_size_padded_x]
                };

                amr::containers::static_vector<double, NVAR> U_ym2 = {
                    rho_patch[linear_idx - 2 * patch_size_padded_x],
                    rhou_patch[linear_idx - 2 * patch_size_padded_x],
                    rhov_patch[linear_idx - 2 * patch_size_padded_x],
                    e_patch[linear_idx - 2 * patch_size_padded_x]
                };

                amr::containers::static_vector<double, NVAR> U_yp1 = {
                    rho_patch[linear_idx + patch_size_padded_x],
                    rhou_patch[linear_idx + patch_size_padded_x],
                    rhov_patch[linear_idx + patch_size_padded_x],
                    e_patch[linear_idx + patch_size_padded_x]
                };

                // Reconstruction (j+1/2)
                reconstructMUSCL(
                    U_ym1,   
                    U_cell,
                    U_yp1,  
                    UL,                
                    UR              
                );

                EulerPhysicsSolver::rusanovFlux(UL, UR, Fy_plus, Direction::Y, gamma);

                // Reconstruction (j-1/2)
                reconstructMUSCL(
                    U_ym2,   
                    U_ym1,
                    U_cell,  
                    UL,                
                    UR              
                );

                EulerPhysicsSolver::rusanovFlux(UL, UR, Fy_minus, Direction::Y, gamma);
                
                // Apply the finite volume update to the temporary buffer
                for (int k = 0; k < NVAR; ++k) {
                    U_new_patch_buffer[linear_idx][k] = U_cell[k] - (dt / dx) * (Fx_plus[k] - Fx_minus[k])
                                            - (dt / dy) * (Fy_plus[k] - Fy_minus[k]);
                }

            } // end cell loop

            // --- Write Back New State (Using Direct Patch Overwrite) ---
            for (std::size_t linear_idx = 0; linear_idx != patch_flat_size; ++linear_idx) {
                if (amr::ndt::utils::patches::is_halo_cell<PatchLayoutT>(linear_idx)) {
                    continue;
                }
                rho_patch[linear_idx]  = U_new_patch_buffer[linear_idx][0];
                rhou_patch[linear_idx]  = U_new_patch_buffer[linear_idx][1];
                rhov_patch[linear_idx]  = U_new_patch_buffer[linear_idx][2];
                e_patch[linear_idx]  = U_new_patch_buffer[linear_idx][3];
            }

        } // end patch loop
    }

    void time_step_3d(double dt){
        constexpr std::size_t patch_size_padded_x = PatchLayoutT::padded_layout_t::shape_t::sizes()[2];
        constexpr std::size_t patch_size_padded_y = PatchLayoutT::padded_layout_t::shape_t::sizes()[1];
        constexpr size_t patch_flat_size = PatchLayoutT::flat_size();
        constexpr auto max_depth = PatchIndexT::max_depth();
        // A temporary buffer to store the new states before committing them to the tree
        std::vector<std::vector<amr::cell::EulerCell3D>> U_new_patches;

        // Loop over the tree and apply the finite volume update
        for (std::size_t patch_idx = 0; patch_idx < m_tree.size(); ++patch_idx) {
            
            // Get current patch's conservative states
            auto& rho_patch = m_tree.template get_patch<amr::cell::Rho>(patch_idx);
            auto& rhou_patch = m_tree.template get_patch<amr::cell::Rhou>(patch_idx);
            auto& rhov_patch = m_tree.template get_patch<amr::cell::Rhov>(patch_idx);
            auto& rhow_patch = m_tree.template get_patch<amr::cell::Rhow>(patch_idx);
            auto& e_patch = m_tree.template get_patch<amr::cell::E3D>(patch_idx);

            // temporary buffer for the new state of this patch
            std::vector<std::array<double, NVAR>> U_new_patch_buffer(patch_flat_size);

            // Loop over cells of patch
            for (std::size_t linear_idx = 0; linear_idx != patch_flat_size;
                ++linear_idx)
            {
                // Skip halo cells
                if (amr::ndt::utils::patches::is_halo_cell<PatchLayoutT>(linear_idx))
                {
                    continue;
                }

                amr::containers::static_vector<double, NVAR> UL, UR;

                // Extract the current conservative state U_cell
                amr::containers::static_vector<double, NVAR> U_cell = {
                    rho_patch[linear_idx],
                    rhou_patch[linear_idx],
                    rhov_patch[linear_idx],
                    rhow_patch[linear_idx],
                    e_patch[linear_idx]
                };

                // Placeholder for fluxes on each face
                amr::containers::static_vector<double, NVAR> Fx_minus;
                amr::containers::static_vector<double, NVAR> Fx_plus;
                amr::containers::static_vector<double, NVAR> Fy_plus;
                amr::containers::static_vector<double, NVAR> Fy_minus;
                amr::containers::static_vector<double, NVAR> Fz_minus;
                amr::containers::static_vector<double, NVAR> Fz_plus;

                // --- Calculate fluxes for the X-direction ---
                amr::containers::static_vector<double, NVAR> U_xp1 = {
                    rho_patch[linear_idx + 1],
                    rhou_patch[linear_idx + 1],
                    rhov_patch[linear_idx + 1],
                    rhow_patch[linear_idx + 1],
                    e_patch[linear_idx + 1]
                };
                
                amr::containers::static_vector<double, NVAR> U_xm1 = {
                    rho_patch[linear_idx - 1],
                    rhou_patch[linear_idx - 1],
                    rhov_patch[linear_idx - 1],
                    rhow_patch[linear_idx - 1],
                    e_patch[linear_idx - 1]
                };
      
                amr::containers::static_vector<double, NVAR> U_xm2 = {
                    rho_patch[linear_idx - 2],
                    rhou_patch[linear_idx - 2],
                    rhov_patch[linear_idx - 2],
                    rhow_patch[linear_idx - 2],
                    e_patch[linear_idx - 2]
                };

                // Reconstruction (i+1/2)
                reconstructMUSCL(
                    U_xm1,   
                    U_cell,         
                    U_xp1,  
                    UL, 
                    UR
                );

                EulerPhysicsSolver::rusanovFlux(UL, UR, Fx_plus, Direction::X, gamma);

                // Reconstruction (i-1/2)
                reconstructMUSCL(
                    U_xm2,   
                    U_xm1,         
                    U_cell,  
                    UL, 
                    UR
                );

                EulerPhysicsSolver::rusanovFlux(UL, UR, Fx_minus, Direction::X, gamma);

                // --- Calculate fluxes for the Y-direction ---
                amr::containers::static_vector<double, NVAR> U_ym1 = {
                    rho_patch[linear_idx - patch_size_padded_x],
                    rhou_patch[linear_idx - patch_size_padded_x],
                    rhov_patch[linear_idx - patch_size_padded_x],
                    rhow_patch[linear_idx - patch_size_padded_x],
                    e_patch[linear_idx - patch_size_padded_x]
                };

                amr::containers::static_vector<double, NVAR> U_ym2 = {
                    rho_patch[linear_idx - 2 * patch_size_padded_x],
                    rhou_patch[linear_idx - 2 * patch_size_padded_x],
                    rhov_patch[linear_idx - 2 * patch_size_padded_x],
                    rhow_patch[linear_idx - 2 * patch_size_padded_x],
                    e_patch[linear_idx - 2 * patch_size_padded_x]
                };

                amr::containers::static_vector<double, NVAR> U_yp1 = {
                    rho_patch[linear_idx + patch_size_padded_x],
                    rhou_patch[linear_idx + patch_size_padded_x],
                    rhov_patch[linear_idx + patch_size_padded_x],
                    rhow_patch[linear_idx + patch_size_padded_x],
                    e_patch[linear_idx + patch_size_padded_x]
                };
                
                // Reconstruction (j+1/2)
                reconstructMUSCL(
                    U_ym1,   
                    U_cell,         
                    U_yp1,  
                    UL, 
                    UR
                );

                EulerPhysicsSolver::rusanovFlux(UL, UR, Fy_plus, Direction::Y, gamma);

                // Reconstruction (j-1/2)
                reconstructMUSCL(
                    U_ym2,   
                    U_ym1,         
                    U_cell,  
                    UL, 
                    UR
                );

                EulerPhysicsSolver::rusanovFlux(UL, UR, Fy_minus, Direction::Y, gamma);

                // --- Calculate fluxes for the Z-direction ---
                amr::containers::static_vector<double, NVAR> U_zp1 = {
                    rho_patch[linear_idx + patch_size_padded_x * patch_size_padded_y],
                    rhou_patch[linear_idx + patch_size_padded_x * patch_size_padded_y],
                    rhov_patch[linear_idx + patch_size_padded_x * patch_size_padded_y],
                    rhow_patch[linear_idx + patch_size_padded_x * patch_size_padded_y],
                    e_patch[linear_idx + patch_size_padded_x * patch_size_padded_y]
                };

                amr::containers::static_vector<double, NVAR> U_zm1 = {
                    rho_patch[linear_idx - patch_size_padded_x * patch_size_padded_y],
                    rhou_patch[linear_idx - patch_size_padded_x * patch_size_padded_y],
                    rhov_patch[linear_idx - patch_size_padded_x * patch_size_padded_y],
                    rhow_patch[linear_idx - patch_size_padded_x * patch_size_padded_y],
                    e_patch[linear_idx - patch_size_padded_x * patch_size_padded_y]
                };

                amr::containers::static_vector<double, NVAR> U_zm2 = {
                    rho_patch[linear_idx - 2 * patch_size_padded_x * patch_size_padded_y],
                    rhou_patch[linear_idx - 2 * patch_size_padded_x * patch_size_padded_y],
                    rhov_patch[linear_idx - 2 * patch_size_padded_x * patch_size_padded_y],
                    rhow_patch[linear_idx - 2 * patch_size_padded_x * patch_size_padded_y],
                    e_patch[linear_idx - 2 * patch_size_padded_x * patch_size_padded_y]
                };
                
                // Reconstruction (k+1/2)
                reconstructMUSCL(
                    U_zm1,   
                    U_cell,         
                    U_zp1,  
                    UL, 
                    UR
                );

                EulerPhysicsSolver::rusanovFlux(UL, UR, Fz_plus, Direction::Z, gamma);

                // Reconstruction (k-1/2)
                reconstructMUSCL(
                    U_zm2,   
                    U_zm1,         
                    U_cell,  
                    UL, 
                    UR
                );

                EulerPhysicsSolver::rusanovFlux(UL, UR, Fz_minus, Direction::Z, gamma);
                
  
                // Apply the finite volume update to the temporary buffer
                auto patch_id = m_tree.get_node_index_at(patch_idx);
                auto level = patch_id.level();
                // auto max_depth = decltype(patch_id)::max_depth();
                uint32_t cell_size = 1u << (max_depth - level);

                double dx = cell_size;
                double dy = cell_size;
                double dz = cell_size;
                
                for (int k = 0; k < NVAR; ++k) {
                    U_new_patch_buffer[linear_idx][k] = U_cell[k] - (dt / dx) * (Fx_plus[k] - Fx_minus[k])
                                            - (dt / dy) * (Fy_plus[k] - Fy_minus[k])
                                            - (dt / dz) * (Fz_plus[k] - Fz_minus[k]);
                }

            } // end cell loop

            // --- Write Back New State (Using Direct Patch Overwrite) ---
            for (std::size_t linear_idx = 0; linear_idx != patch_flat_size; ++linear_idx) {
                if (amr::ndt::utils::patches::is_halo_cell<PatchLayoutT>(linear_idx)) {
                    continue;
                }
                rho_patch[linear_idx]  = U_new_patch_buffer[linear_idx][0];
                rhou_patch[linear_idx]  = U_new_patch_buffer[linear_idx][1];
                rhov_patch[linear_idx]  = U_new_patch_buffer[linear_idx][2];
                rhow_patch[linear_idx] = U_new_patch_buffer[linear_idx][3];
                e_patch[linear_idx]    = U_new_patch_buffer[linear_idx][4];
            }

        } // end patch loop
    }

    double compute_time_step_2d() const {
        constexpr auto patch_flat_size = PatchLayoutT::flat_size();
        
        double dt_min = 1e10;

        // Loop over all patches
        for (std::size_t patch_idx = 0; patch_idx < m_tree.size(); ++patch_idx) {
            
            auto patch_id = m_tree.get_node_index_at(patch_idx);
            
            // Get cell sizes using physics system
            auto cell_size = physics_system_t::cell_sizes(patch_id);
            double dx = cell_size[0];
            double dy = cell_size[1];

            // Get current patch's conservative states
            const auto& rho_patch = m_tree.template get_patch<amr::cell::Rho>(patch_idx);
            const auto& rhou_patch = m_tree.template get_patch<amr::cell::Rhou>(patch_idx);
            const auto& rhov_patch = m_tree.template get_patch<amr::cell::Rhov>(patch_idx);
            const auto& e_patch = m_tree.template get_patch<amr::cell::E2D>(patch_idx);

            // Loop over cells
            for (std::size_t linear_idx = 0; linear_idx != patch_flat_size; ++linear_idx) {
                
                if (amr::ndt::utils::patches::is_halo_cell<PatchLayoutT>(linear_idx)) {
                    continue;
                }

                // Convert to primitive
                amr::containers::static_vector<double, NVAR> U_cell = {
                    rho_patch[linear_idx],
                    rhou_patch[linear_idx],
                    rhov_patch[linear_idx],
                    e_patch[linear_idx]
                };
                amr::containers::static_vector<double, NVAR> prim;
                EulerPhysicsSolver::conservativeToPrimitive(U_cell, prim, gamma);

                // Get sound speed
                double a = EulerPhysicsSolver::computeSoundSpeed(U_cell, gamma);

                // Compute time step for X direction
                double u_x = prim[1];
                double dt_x = dx / (std::abs(u_x) + a);
                dt_min = std::min(dt_min, dt_x);
                
                // Compute time step for Y direction
                double u_y = prim[2];
                double dt_y = dy / (std::abs(u_y) + a);
                dt_min = std::min(dt_min, dt_y);
            }
        }

        return cfl * dt_min;
    }

    double compute_time_step_3d() const {
        constexpr size_t patch_flat_size = PatchLayoutT::flat_size();
        constexpr auto max_depth = PatchIndexT::max_depth();
        
        double dt_min = 1e10;

        // Loop over all patches
        for (std::size_t patch_idx = 0; patch_idx < m_tree.size(); ++patch_idx) {
            
            // Get patch properties
            auto patch_id = m_tree.get_node_index_at(patch_idx);
            auto level = patch_id.level();
            uint32_t cell_size = 1u << (max_depth - level);
            double dx = cell_size;

            // Get current patch's conservative states
            const auto& rho_patch = m_tree.template get_patch<amr::cell::Rho>(patch_idx);
            const auto& rhou_patch = m_tree.template get_patch<amr::cell::Rhou>(patch_idx);
            const auto& rhov_patch = m_tree.template get_patch<amr::cell::Rhov>(patch_idx);
            const auto& rhow_patch = m_tree.template get_patch<amr::cell::Rhow>(patch_idx);
            const auto& e_patch = m_tree.template get_patch<amr::cell::E3D>(patch_idx);

            // Loop over cells
            for (std::size_t linear_idx = 0; linear_idx != patch_flat_size; ++linear_idx) {
                
                if (amr::ndt::utils::patches::is_halo_cell<PatchLayoutT>(linear_idx)) {
                    continue;
                }

                // Convert to primitive
                amr::containers::static_vector<double, NVAR> U_cell = {
                    rho_patch[linear_idx],
                    rhou_patch[linear_idx],
                    rhov_patch[linear_idx],
                    rhow_patch[linear_idx],
                    e_patch[linear_idx]
                };
                amr::containers::static_vector<double, NVAR> prim;
                EulerPhysicsSolver::conservativeToPrimitive(U_cell, prim, gamma);

                // Get sound speed
                double a = EulerPhysicsSolver::computeSoundSpeed(U_cell, gamma);

                // Compute time step for each direction
                for (int dim = 0; dim < 3; ++dim) {
                    double u_dim = prim[1 + dim];
                    double dt_dir = dx / (std::abs(u_dim) + a);
                    dt_min = std::min(dt_min, dt_dir);
                }
            }
        }

        return cfl * dt_min;
    }

    template<int NVAR>
    static void reconstructMUSCL(
        const amr::containers::static_vector<double, NVAR>& Um,
        const amr::containers::static_vector<double, NVAR>& U0,
        const amr::containers::static_vector<double, NVAR>& Up,
        amr::containers::static_vector<double, NVAR>& UL,
        amr::containers::static_vector<double, NVAR>& UR
    ) {
        for (int k = 0; k < NVAR; ++k) {
            double dL = U0[k] - Um[k];
            double dR = Up[k] - U0[k];
            double slope = minmod(dL, dR);

            UL[k] = U0[k] + 0.5 * slope;
            UR[k] = Up[k] - 0.5 * slope;
        }
    }

    static double minmod(double a, double b) {
    if (a * b <= 0.0) return 0.0;
    return (std::abs(a) < std::abs(b)) ? a : b;
    }
};

// Typedefs
template<typename TreeT, typename PhysicsT>
using amr_solver_2d = amr_solver_MUSCL<TreeT, PhysicsT, 2>;

template<typename TreeT, typename PhysicsT>
using amr_solver_3d = amr_solver_MUSCL<TreeT, PhysicsT, 3>;

#endif // AMR_SOLVER_MUSCL_HPP