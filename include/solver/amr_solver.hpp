#ifndef AMR_SOLVER_HPP
#define AMR_SOLVER_HPP

#include "ndtree/ndtree.hpp"
#include "cell_types.hpp"
#include "EulerSolver2D.hpp"

#include <vector>
#include <functional>

template<typename TreeT>
class amr_solver {
private:
    TreeT m_tree;
    EulerSolver2D m_euler_solver;

public:
    using PatchLayoutT = typename TreeT::patch_layout_t;
    using PatchIndexT = typename TreeT::patch_index_t;

    amr_solver(size_t capacity, double gamma = 1.4): m_tree(capacity), m_euler_solver(1, 1, 1.0, 1.0, gamma, 0.3) {
        // EulerSolver2D constructor needs dummy grid dimensions, dx, dy for a single point.
        // It's not used directly for the time step but is necessary to instantiate.
    }

    TreeT& get_tree() {
        return m_tree;
    }

    void initialize(std::function<std::vector<double>(double, double)> init_func) {
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
            
            // Size of a single cell in terms of the maximum coordinate units (e.g., 64 for level 6)
            uint32_t cell_size_units = 1u << (max_depth - level);
            
            // Get references to the patch data buffers
            auto& rho_patch  = m_tree.template get_patch<amr::cell::Rho>(patch_idx);
            auto& rhou_patch = m_tree.template get_patch<amr::cell::Rhou>(patch_idx);
            auto& rhov_patch = m_tree.template get_patch<amr::cell::Rhov>(patch_idx);
            auto& e_patch    = m_tree.template get_patch<amr::cell::E>(patch_idx);

            // Loop over cells in the patch and set IC
            for (std::size_t linear_idx = 0; linear_idx != patch_flat_size; ++linear_idx) {
                
                // Skip Halo Cells on initialization
                //if (amr::ndt::utils::patches::is_halo_cell<PatchLayoutT>(linear_idx)) {
                //    continue;
                //}

                // Calculate 2D coordinates (i, j) within the **unpadded** patch layout
                uint32_t i = static_cast<uint32_t>(linear_idx % patch_size_padded_x);
                uint32_t j = static_cast<uint32_t>(linear_idx / patch_size_padded_x);
                
                // Calculate absolute cell center coordinates in max_coord units
                // x_units = (Patch Origin X) + (Cell Index i) * (Cell Size in units) + (Half Cell Size)
                uint32_t cell_x_units = patch_coords_raw[0] * cell_size_units + 
                                        static_cast<uint32_t>(i) * cell_size_units + 
                                        cell_size_units / 2;
                
                uint32_t cell_y_units = patch_coords_raw[1] * cell_size_units + 
                                        static_cast<uint32_t>(j) * cell_size_units + 
                                        cell_size_units / 2;

                // Convert to normalized world coordinates [0, 1]
                double x_world = static_cast<double>(cell_x_units) / max_coord_x;
                double y_world = static_cast<double>(cell_y_units) / max_coord_y;


                // Evaluate the Initial Condition
                std::vector<double> prim = init_func(x_world, y_world);

                // Convert to conservative and write to patch
                std::vector<double> cons(4);
                m_euler_solver.primitiveToConservative(prim, cons);
                
                rho_patch[linear_idx]  = cons[0];
                rhou_patch[linear_idx] = cons[1];
                rhov_patch[linear_idx] = cons[2];
                e_patch[linear_idx]    = cons[3];
            }
        }
        // After initialization, the halo regions must be synchronized for the first flux calculation
        // m_tree.exchange_halos();

    }

    void time_step(double dt){
        constexpr size_t patch_size_padded_x = PatchLayoutT::padded_layout_t::shape_t::sizes()[1];
        constexpr size_t patch_flat_size = PatchLayoutT::flat_size();
        constexpr auto max_depth = PatchIndexT::max_depth();
        // A temporary buffer to store the new states before committing them to the tree
        std::vector<std::vector<amr::cell::EulerCell>> U_new_patches;

        // Loop over the tree and apply the finite volume update
        for (std::size_t patch_idx = 0; patch_idx < m_tree.size(); ++patch_idx) {
            
            // Get current patch's conservative states
            auto& rho_patch = m_tree.template get_patch<amr::cell::Rho>(patch_idx);
            auto& rhou_patch = m_tree.template get_patch<amr::cell::Rhou>(patch_idx);
            auto& rhov_patch = m_tree.template get_patch<amr::cell::Rhov>(patch_idx);
            auto& e_patch = m_tree.template get_patch<amr::cell::E>(patch_idx);

            // temporary buffer for the new state of this patch
            std::vector<std::array<double, 4>> U_new_patch_buffer(patch_flat_size);

            // Loop over cells of patch
            for (std::size_t linear_idx = 0; linear_idx != patch_flat_size;
                ++linear_idx)
            {
                if (amr::ndt::utils::patches::is_halo_cell<PatchLayoutT>(linear_idx))
                {
                    continue;
                }

                // Extract the current conservative state U_cell
                std::vector<double> U_cell = {
                    rho_patch[linear_idx],
                    rhou_patch[linear_idx],
                    rhov_patch[linear_idx],
                    e_patch[linear_idx]
                };

                // Placeholder for fluxes on each face
                std::vector<double> flux_left(4, 0.0);
                std::vector<double> flux_right(4, 0.0);
                std::vector<double> flux_bottom(4, 0.0);
                std::vector<double> flux_top(4, 0.0);

                // --- Calculate fluxes for the right face (X-direction) ---
                std::vector<double> U_right_neighbor = {
                    rho_patch[linear_idx + 1],
                    rhou_patch[linear_idx + 1],
                    rhov_patch[linear_idx + 1],
                    e_patch[linear_idx + 1]
                };
                m_euler_solver.rusanovFlux(U_cell, U_right_neighbor, flux_right, Direction::X);
                
                // --- Calculate fluxes for the left face (X-direction) ---
                std::vector<double> U_left_neighbor = {
                    rho_patch[linear_idx - 1],
                    rhou_patch[linear_idx - 1],
                    rhov_patch[linear_idx - 1],
                    e_patch[linear_idx - 1]
                };
                m_euler_solver.rusanovFlux(U_left_neighbor, U_cell, flux_left, Direction::X);

                // --- Calculate fluxes for the top face (Y-direction) ---
                std::vector<double> U_top_neighbor = {
                    rho_patch[linear_idx - patch_size_padded_x],
                    rhou_patch[linear_idx - patch_size_padded_x],
                    rhov_patch[linear_idx - patch_size_padded_x],
                    e_patch[linear_idx - patch_size_padded_x]
                };
                m_euler_solver.rusanovFlux(U_cell, U_top_neighbor, flux_top, Direction::Y);

                // --- Calculate fluxes for the bottom face (Y-direction) --- 
                std::vector<double> U_bottom_neighbor = {
                    rho_patch[linear_idx + patch_size_padded_x],
                    rhou_patch[linear_idx + patch_size_padded_x],
                    rhov_patch[linear_idx + patch_size_padded_x],
                    e_patch[linear_idx + patch_size_padded_x]
                };
                m_euler_solver.rusanovFlux(U_bottom_neighbor, U_cell, flux_bottom, Direction::Y);
                
                
                // Apply the finite volume update to the temporary buffer
                auto patch_id = m_tree.get_node_index_at(patch_idx);
                auto level = patch_id.level();
                // auto max_depth = decltype(patch_id)::max_depth();
                uint32_t cell_size = 1u << (max_depth - level);

                double dx = cell_size;
                double dy = cell_size;
                
                for (size_t k = 0; k < 4; ++k) {
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

};

#endif // AMR_SOLVER_HPP