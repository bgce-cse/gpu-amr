#ifndef AMR_SOLVER_HPP
#define AMR_SOLVER_HPP

#include "ndtree/ndtree.hpp"
#include "cell_types.hpp"
#include "EulerPhysics.hpp"
#include "physics_system.hpp"
#include <vector>
#include <functional>

template<typename TreeT, typename PhysicsT, int DIM>
class amr_solver {
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

    amr_solver(size_t capacity, double gamma_ = 1.4, double cfl_ = 0.1)
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
        constexpr std::size_t patch_flat_size = PatchLayoutT::flat_size();

        // Loop over all patches
        for (std::size_t patch_idx = 0; patch_idx < m_tree.size(); ++patch_idx) {
            
            auto patch_id = m_tree.get_node_index_at(patch_idx);
            
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

                // Get cell center coordinates using physics system
                auto cell_center = physics_system_t::cell_coord(patch_id, linear_idx);
                
                // For cell centers, add 0.5 * cell_size
                auto cell_size = physics_system_t::cell_sizes(patch_id);
                double x_world = cell_center[0] + 0.5 * cell_size[0];
                double y_world = cell_center[1] + 0.5 * cell_size[1];
                double z_world = cell_center[2] + 0.5 * cell_size[2];

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

        for (std::size_t patch_idx = 0; patch_idx < m_tree.size(); ++patch_idx) {
            
            auto patch_id = m_tree.get_node_index_at(patch_idx);
            
            auto cell_size = physics_system_t::cell_sizes(patch_id);
            double dx = cell_size[0];
            double dy = cell_size[1];
            double dz = cell_size[2];
            
            auto& rho_patch = m_tree.template get_patch<amr::cell::Rho>(patch_idx);
            auto& rhou_patch = m_tree.template get_patch<amr::cell::Rhou>(patch_idx);
            auto& rhov_patch = m_tree.template get_patch<amr::cell::Rhov>(patch_idx);
            auto& rhow_patch = m_tree.template get_patch<amr::cell::Rhow>(patch_idx);
            auto& e_patch = m_tree.template get_patch<amr::cell::E3D>(patch_idx);

            std::vector<std::array<double, NVAR>> U_new_patch_buffer(patch_flat_size);

            for (std::size_t linear_idx = 0; linear_idx != patch_flat_size; ++linear_idx)
            {
                if (amr::ndt::utils::patches::is_halo_cell<PatchLayoutT>(linear_idx))
                {
                    continue;
                }

                amr::containers::static_vector<double, NVAR> U_cell = {
                    rho_patch[linear_idx],
                    rhou_patch[linear_idx],
                    rhov_patch[linear_idx],
                    rhow_patch[linear_idx],
                    e_patch[linear_idx]
                };

                amr::containers::static_vector<double, NVAR> flux_left, flux_right;
                amr::containers::static_vector<double, NVAR> flux_bottom, flux_top;
                amr::containers::static_vector<double, NVAR> flux_back, flux_front;

                // X-direction: RIGHT face
                amr::containers::static_vector<double, NVAR> U_right = {
                    rho_patch[linear_idx + 1],
                    rhou_patch[linear_idx + 1],
                    rhov_patch[linear_idx + 1],
                    rhow_patch[linear_idx + 1],
                    e_patch[linear_idx + 1]
                };
                EulerPhysicsSolver::rusanovFlux(U_cell, U_right, flux_right, Direction::X, gamma);
                
                // X-direction: LEFT face
                amr::containers::static_vector<double, NVAR> U_left = {
                    rho_patch[linear_idx - 1],
                    rhou_patch[linear_idx - 1],
                    rhov_patch[linear_idx - 1],
                    rhow_patch[linear_idx - 1],
                    e_patch[linear_idx - 1]
                };
                EulerPhysicsSolver::rusanovFlux(U_left, U_cell, flux_left, Direction::X, gamma);

                // Y-direction: TOP face (- padded_x)
                amr::containers::static_vector<double, NVAR> U_top = {
                    rho_patch[linear_idx - patch_size_padded_x],
                    rhou_patch[linear_idx - patch_size_padded_x],
                    rhov_patch[linear_idx - patch_size_padded_x],
                    rhow_patch[linear_idx - patch_size_padded_x],
                    e_patch[linear_idx - patch_size_padded_x]
                };
                EulerPhysicsSolver::rusanovFlux(U_top, U_cell, flux_top, Direction::Y, gamma);

                // Y-direction: BOTTOM face (+ padded_x)
                amr::containers::static_vector<double, NVAR> U_bottom = {
                    rho_patch[linear_idx + patch_size_padded_x],
                    rhou_patch[linear_idx + patch_size_padded_x],
                    rhov_patch[linear_idx + patch_size_padded_x],
                    rhow_patch[linear_idx + patch_size_padded_x],
                    e_patch[linear_idx + patch_size_padded_x]
                };
                EulerPhysicsSolver::rusanovFlux(U_cell, U_bottom, flux_bottom, Direction::Y, gamma);

                // Z-direction: BACK face (- padded_x * padded_y)
                amr::containers::static_vector<double, NVAR> U_back = {
                    rho_patch[linear_idx - patch_size_padded_x * patch_size_padded_y],
                    rhou_patch[linear_idx - patch_size_padded_x * patch_size_padded_y],
                    rhov_patch[linear_idx - patch_size_padded_x * patch_size_padded_y],
                    rhow_patch[linear_idx - patch_size_padded_x * patch_size_padded_y],
                    e_patch[linear_idx - patch_size_padded_x * patch_size_padded_y]
                };
                EulerPhysicsSolver::rusanovFlux(U_back, U_cell, flux_back, Direction::Z, gamma);

                // Z-direction: FRONT face (+ padded_x * padded_y)
                amr::containers::static_vector<double, NVAR> U_front = {
                    rho_patch[linear_idx + patch_size_padded_x * patch_size_padded_y],
                    rhou_patch[linear_idx + patch_size_padded_x * patch_size_padded_y],
                    rhov_patch[linear_idx + patch_size_padded_x * patch_size_padded_y],
                    rhow_patch[linear_idx + patch_size_padded_x * patch_size_padded_y],
                    e_patch[linear_idx + patch_size_padded_x * patch_size_padded_y]
                };
                EulerPhysicsSolver::rusanovFlux(U_cell, U_front, flux_front, Direction::Z, gamma);
                
                // Finite volume update
                for (int k = 0; k < NVAR; ++k) {
                    U_new_patch_buffer[linear_idx][k] = U_cell[k] 
                        - (dt / dx) * (flux_right[k] - flux_left[k])
                        - (dt / dy) * (flux_bottom[k] - flux_top[k])
                        - (dt / dz) * (flux_front[k] - flux_back[k]);
                }
            }

            // Write back
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
        }
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
        
        double dt_min = 1e10;

        // Loop over all patches
        for (std::size_t patch_idx = 0; patch_idx < m_tree.size(); ++patch_idx) {
            
            auto patch_id = m_tree.get_node_index_at(patch_idx);
            
            // Get cell sizes using physics system
            auto cell_size = physics_system_t::cell_sizes(patch_id);
            double dx = cell_size[0];
            double dy = cell_size[1];
            double dz = cell_size[2];

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

                // Compute time step for X direction
                double u_x = prim[1];
                double dt_x = dx / (std::abs(u_x) + a);
                dt_min = std::min(dt_min, dt_x);
                
                // Compute time step for Y direction
                double u_y = prim[2];
                double dt_y = dy / (std::abs(u_y) + a);
                dt_min = std::min(dt_min, dt_y);
                
                // Compute time step for Z direction
                double u_z = prim[3];
                double dt_z = dz / (std::abs(u_z) + a);
                dt_min = std::min(dt_min, dt_z);
            }
        }

        return cfl * dt_min;
    }
};

// Typedefs
template<typename TreeT, typename PhysicsT>
using amr_solver_2d = amr_solver<TreeT, PhysicsT, 2>;

template<typename TreeT, typename PhysicsT>
using amr_solver_3d = amr_solver<TreeT, PhysicsT, 3>;

#endif // AMR_SOLVER_HPP