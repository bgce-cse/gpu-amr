#ifndef AMR_SOLVER_HPP
#define AMR_SOLVER_HPP

#include "ndtree/ndtree.hpp"
#include "cell_types.hpp"
#include "EulerPhysics.hpp"

#include <vector>
#include <functional>

template<typename TreeT, int DIM>
class amr_solver {
private:
    TreeT m_tree;
    double gamma;       // Specific heat ratio
    double cfl;         // CFL number

public:
    using PatchLayoutT = typename TreeT::patch_layout_t;
    using PatchIndexT = typename TreeT::patch_index_t;
    using EulerPhysicsSolver = EulerPhysics<DIM>;
    static constexpr int NVAR = EulerPhysicsSolver::NVAR;

    amr_solver(size_t capacity, double gamma_ = 1.4, double cfl_ = 0.3)
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
        // Get geometrical constants
        // constexpr double global_size = 1.0;
        //constexpr size_t patch_size_x = Patch_Layout::data_layout_t::sizes()[0]; 
        constexpr std::size_t patch_size_padded_x = PatchLayoutT::padded_layout_t::shape_t::sizes()[1];
        constexpr std::size_t patch_size_padded_y = PatchLayoutT::padded_layout_t::shape_t::sizes()[0];
        constexpr std::size_t patch_flat_size = PatchLayoutT::flat_size();

        // Get max_depth from the TreeT definition
        constexpr auto max_depth = PatchIndexT::max_depth();

        // The total coordinate range is 2^max_depth (e.g., 512 for max_depth=9)
        constexpr double max_cell_size = 1u << max_depth;
        constexpr double max_coord_x = max_cell_size * patch_size_padded_x;
        constexpr double max_coord_y = max_cell_size * patch_size_padded_y;

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
            auto& e_patch    = m_tree.template get_patch<amr::cell::E2D>(patch_idx);

            // Loop over cells in the patch and set IC
            for (std::size_t linear_idx = 0; linear_idx != patch_flat_size; ++linear_idx) {
                
                // Skip Halo Cells on initialization
                if (amr::ndt::utils::patches::is_halo_cell<PatchLayoutT>(linear_idx)) {
                    continue;
                }

                // Calculate 2D coordinates (i, j) within the **unpadded** patch layout
                uint32_t i = static_cast<uint32_t>(linear_idx % patch_size_padded_x);
                uint32_t j = static_cast<uint32_t>(linear_idx / patch_size_padded_x);
                
                // Calculate absolute cell center coordinates in max_coord units
                // x_units = (Patch Origin X) + (Cell Index i) * (Cell Size in units) + (Half Cell Size)
                uint32_t cell_x_units = patch_coords_raw[0] *  static_cast<uint32_t>(patch_size_padded_x) + 
                                        static_cast<uint32_t>(i) * cell_size_units + 
                                        cell_size_units / 2;
                
                uint32_t cell_y_units = patch_coords_raw[1] *  static_cast<uint32_t>(patch_size_padded_y) + 
                                        static_cast<uint32_t>(j) * cell_size_units + 
                                        cell_size_units / 2;

                // Convert to normalized world coordinates [0, 1]
                double x_world = static_cast<double>(cell_x_units) / max_coord_x;
                double y_world = static_cast<double>(cell_y_units) / max_coord_y;


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
        constexpr auto max_depth = PatchIndexT::max_depth();
        // A temporary buffer to store the new states before committing them to the tree
        std::vector<std::vector<amr::cell::EulerCell2D>> U_new_patches;

        // Loop over the tree and apply the finite volume update
        for (std::size_t patch_idx = 0; patch_idx < m_tree.size(); ++patch_idx) {
            
            // Get current patch's conservative states
            auto& rho_patch = m_tree.template get_patch<amr::cell::Rho>(patch_idx);
            auto& rhou_patch = m_tree.template get_patch<amr::cell::Rhou>(patch_idx);
            auto& rhov_patch = m_tree.template get_patch<amr::cell::Rhov>(patch_idx);
            auto& e_patch = m_tree.template get_patch<amr::cell::E2D>(patch_idx);

            // temporary buffer for the new state of this patch
            std::vector<std::array<double, NVAR>> U_new_patch_buffer(patch_flat_size);

            // Loop over cells of patch
            for (std::size_t linear_idx = 0; linear_idx != patch_flat_size;
                ++linear_idx)
            {
                // SKip halo cells
                if (amr::ndt::utils::patches::is_halo_cell<PatchLayoutT>(linear_idx))
                {
                    continue;
                }

                // Extract the current conservative state U_cell
                amr::containers::static_vector<double, NVAR> U_cell = {
                    rho_patch[linear_idx],
                    rhou_patch[linear_idx],
                    rhov_patch[linear_idx],
                    e_patch[linear_idx]
                };

                // Placeholder for fluxes on each face
                amr::containers::static_vector<double, NVAR> flux_left;
                amr::containers::static_vector<double, NVAR> flux_right;
                amr::containers::static_vector<double, NVAR> flux_bottom;
                amr::containers::static_vector<double, NVAR> flux_top;

                // --- Calculate fluxes for the right face (X-direction) ---
                amr::containers::static_vector<double, NVAR> U_right_neighbor = {
                    rho_patch[linear_idx + 1],
                    rhou_patch[linear_idx + 1],
                    rhov_patch[linear_idx + 1],
                    e_patch[linear_idx + 1]
                };
                EulerPhysicsSolver::rusanovFlux(U_cell, U_right_neighbor, flux_right, Direction::X, gamma);
                
                // --- Calculate fluxes for the left face (X-direction) ---
                amr::containers::static_vector<double, NVAR> U_left_neighbor = {
                    rho_patch[linear_idx - 1],
                    rhou_patch[linear_idx - 1],
                    rhov_patch[linear_idx - 1],
                    e_patch[linear_idx - 1]
                };
                EulerPhysicsSolver::rusanovFlux(U_left_neighbor, U_cell, flux_left, Direction::X, gamma);

                // --- Calculate fluxes for the top face (Y-direction) ---
                amr::containers::static_vector<double, NVAR> U_top_neighbor = {
                    rho_patch[linear_idx - patch_size_padded_x],
                    rhou_patch[linear_idx - patch_size_padded_x],
                    rhov_patch[linear_idx - patch_size_padded_x],
                    e_patch[linear_idx - patch_size_padded_x]
                };
                EulerPhysicsSolver::rusanovFlux(U_cell, U_top_neighbor, flux_top, Direction::Y, gamma);

                // --- Calculate fluxes for the bottom face (Y-direction) --- 
                amr::containers::static_vector<double, NVAR> U_bottom_neighbor = {
                    rho_patch[linear_idx + patch_size_padded_x],
                    rhou_patch[linear_idx + patch_size_padded_x],
                    rhov_patch[linear_idx + patch_size_padded_x],
                    e_patch[linear_idx + patch_size_padded_x]
                };
                EulerPhysicsSolver::rusanovFlux(U_bottom_neighbor, U_cell, flux_bottom, Direction::Y, gamma);
                
                // Apply the finite volume update to the temporary buffer
                auto patch_id = m_tree.get_node_index_at(patch_idx);
                auto level = patch_id.level();
                // auto max_depth = decltype(patch_id)::max_depth();
                uint32_t cell_size = 1u << (max_depth - level);

                double dx = cell_size;
                double dy = cell_size;
                
                for (int k = 0; k < NVAR; ++k) {
                    U_new_patch_buffer[linear_idx][k] = U_cell[k] - (dt / dx) * (flux_right[k] - flux_left[k])
                                            - (dt / dy) * (flux_top[k] - flux_bottom[k]);
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

                // Extract the current conservative state U_cell
                amr::containers::static_vector<double, NVAR> U_cell = {
                    rho_patch[linear_idx],
                    rhou_patch[linear_idx],
                    rhov_patch[linear_idx],
                    rhow_patch[linear_idx],
                    e_patch[linear_idx]
                };

                // Placeholder for fluxes on each face
                amr::containers::static_vector<double, NVAR> flux_left;
                amr::containers::static_vector<double, NVAR> flux_right;
                amr::containers::static_vector<double, NVAR> flux_bottom;
                amr::containers::static_vector<double, NVAR> flux_top;
                amr::containers::static_vector<double, NVAR> flux_back;
                amr::containers::static_vector<double, NVAR> flux_front;

                // --- Calculate fluxes for the right face (X-direction) ---
                amr::containers::static_vector<double, NVAR> U_right_neighbor = {
                    rho_patch[linear_idx + 1],
                    rhou_patch[linear_idx + 1],
                    rhov_patch[linear_idx + 1],
                    rhow_patch[linear_idx + 1],
                    e_patch[linear_idx + 1]
                };
                EulerPhysicsSolver::rusanovFlux(U_cell, U_right_neighbor, flux_right, Direction::X, gamma);
                
                // --- Calculate fluxes for the left face (X-direction) ---
                amr::containers::static_vector<double, NVAR> U_left_neighbor = {
                    rho_patch[linear_idx - 1],
                    rhou_patch[linear_idx - 1],
                    rhov_patch[linear_idx - 1],
                    rhow_patch[linear_idx - 1],
                    e_patch[linear_idx - 1]
                };
                EulerPhysicsSolver::rusanovFlux(U_left_neighbor, U_cell, flux_left, Direction::X, gamma);

                // --- Calculate fluxes for the top face (Y-direction) ---
                amr::containers::static_vector<double, NVAR> U_top_neighbor = {
                    rho_patch[linear_idx - patch_size_padded_x],
                    rhou_patch[linear_idx - patch_size_padded_x],
                    rhov_patch[linear_idx - patch_size_padded_x],
                    rhow_patch[linear_idx - patch_size_padded_x],
                    e_patch[linear_idx - patch_size_padded_x]
                };
                EulerPhysicsSolver::rusanovFlux(U_cell, U_top_neighbor, flux_top, Direction::Y, gamma);

                // --- Calculate fluxes for the bottom face (Y-direction) --- 
                amr::containers::static_vector<double, NVAR> U_bottom_neighbor = {
                    rho_patch[linear_idx + patch_size_padded_x],
                    rhou_patch[linear_idx + patch_size_padded_x],
                    rhov_patch[linear_idx + patch_size_padded_x],
                    rhow_patch[linear_idx + patch_size_padded_x],
                    e_patch[linear_idx + patch_size_padded_x]
                };
                EulerPhysicsSolver::rusanovFlux(U_bottom_neighbor, U_cell, flux_bottom, Direction::Y, gamma);

                // --- Calculate fluxes for the front face (Z-direction) ---
                amr::containers::static_vector<double, NVAR> U_front_neighbor = {
                    rho_patch[linear_idx + patch_size_padded_x * patch_size_padded_y],
                    rhou_patch[linear_idx + patch_size_padded_x * patch_size_padded_y],
                    rhov_patch[linear_idx + patch_size_padded_x * patch_size_padded_y],
                    rhow_patch[linear_idx + patch_size_padded_x * patch_size_padded_y],
                    e_patch[linear_idx + patch_size_padded_x * patch_size_padded_y]
                };
                EulerPhysicsSolver::rusanovFlux(U_front_neighbor, U_cell, flux_front, Direction::Z, gamma);

                // --- Calculate fluxes for the back face (Z-direction) --- 
                amr::containers::static_vector<double, NVAR> U_back_neighbor = {
                    rho_patch[linear_idx - patch_size_padded_x * patch_size_padded_y],
                    rhou_patch[linear_idx - patch_size_padded_x * patch_size_padded_y],
                    rhov_patch[linear_idx - patch_size_padded_x * patch_size_padded_y],
                    rhow_patch[linear_idx - patch_size_padded_x * patch_size_padded_y],
                    e_patch[linear_idx - patch_size_padded_x * patch_size_padded_y]
                };
                EulerPhysicsSolver::rusanovFlux(U_cell, U_back_neighbor, flux_back, Direction::Z, gamma);
                
                
                // Apply the finite volume update to the temporary buffer
                auto patch_id = m_tree.get_node_index_at(patch_idx);
                auto level = patch_id.level();
                // auto max_depth = decltype(patch_id)::max_depth();
                uint32_t cell_size = 1u << (max_depth - level);

                double dx = cell_size;
                double dy = cell_size;
                double dz = cell_size;
                
                for (int k = 0; k < NVAR; ++k) {
                    U_new_patch_buffer[linear_idx][k] = U_cell[k] - (dt / dx) * (flux_right[k] - flux_left[k])
                                            - (dt / dy) * (flux_top[k] - flux_bottom[k])
                                            - (dt / dz) * (flux_front[k] - flux_back[k]);
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

                // Compute time step for each direction
                for (int dim = 0; dim < 2; ++dim) {
                    double u_dim = prim[1 + dim];
                    double dt_dir = dx / (std::abs(u_dim) + a);
                    dt_min = std::min(dt_min, dt_dir);
                }
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
};

// Typedefs
template<typename TreeT>
using amr_solver_2d = amr_solver<TreeT, 2>;

template<typename TreeT>
using amr_solver_3d = amr_solver<TreeT, 3>;

#endif // AMR_SOLVER_HPP