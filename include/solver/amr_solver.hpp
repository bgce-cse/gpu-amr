#ifndef AMR_SOLVER_HPP
#define AMR_SOLVER_HPP

#include "ndtree/ndtree.hpp"
#include "cell_types.hpp"
#include "EulerSolver2D.h"

#include <vector>
#include <functional>

template<typename TreeT>
class amr_solver {
private:
    TreeT m_tree;
    EulerSolver2D m_euler_solver;

public:
    amr_solver(size_t capacity, double gamma = 1.4): m_tree(capacity), m_euler_solver(1, 1, 1.0, 1.0, gamma, 0.3) {
        // EulerSolver2D constructor needs dummy grid dimensions, dx, dy for a single point.
        // It's not used directly for the time step but is necessary to instantiate.
    }

    TreeT& get_tree() {
        return m_tree;
    }

    void initialize(std::function<std::vector<double>(double, double)> init_func) {
        // Get the root node, which is the only node present at the start.
        auto root_node_id = m_tree.get_node_index_at(0);
        
        // Get the level and size of the root node
        auto root_level = root_node_id.level();
        auto max_depth = root_node_id.max_depth();
        double cell_size = 1.0 / (1 << (max_depth - root_level)); // Assuming a domain of [0,1]x[0,1]
        
        // Get the coordinates of the root node (which are 0,0)
        auto [coords, _] = root_node_id.decode(root_node_id.id());
        double x = coords[0] * cell_size;
        double y = coords[1] * cell_size;
        
        // Evaluate the initial condition at the center of the root cell
        double mid_x = x + 0.5 * cell_size;
        double mid_y = y + 0.5 * cell_size;
        std::vector<double> prim = init_func(mid_x, mid_y);
        
        // Convert primitive variables to conservative variables
        std::vector<double> cons(4);
        // You'll need to use your EulerSolver2D's method for this
        m_euler_solver.primitiveToConservative(prim, cons);
        
        // Create the EulerCell and scatter it to the tree
        amr::cell::EulerCell root_cell(cons[0], cons[1], cons[2], cons[3]);
        m_tree.scatter_node(root_cell, 0);
    }

    void time_step(double dt){
        // A temporary buffer to store the new states before committing them to the tree
        std::vector<std::vector<double>> U_new(m_tree.size(), std::vector<double>(4));

        // Loop over the tree and apply the finite volume update
        for (size_t i = 0; i < m_tree.size(); ++i) {
            auto node_id = m_tree.get_node_index_at(i);
            
            // Get current cell's conservative state
            std::vector<double> U_current = {
                m_tree.template get<amr::cell::Rho>(i),
                m_tree.template get<amr::cell::Rhou>(i),
                m_tree.template get<amr::cell::Rhov>(i),
                m_tree.template get<amr::cell::E>(i)
            };
            
            // Placeholder for fluxes on each face
            std::vector<double> flux_left(4, 0.0);
            std::vector<double> flux_right(4, 0.0);
            std::vector<double> flux_bottom(4, 0.0);
            std::vector<double> flux_top(4, 0.0);

            // Calculate fluxes for all four faces (currently simplifying the different kinds of neighbors by just averaging if several are provided -> needs to be checked for correctness)
            // --- Calculate fluxes for the right face (X-direction) ---
            auto neighbors_right = m_tree.get_neighbors(node_id, TreeT::node_index_directon_t::right);
            if (neighbors_right.has_value()) {
                std::vector<double> U_neighbor_avg(4, 0.0);
                for (const auto& neighbor_id : neighbors_right.value()) {
                    auto neighbor_idx_opt = m_tree.find_index(neighbor_id);
                    if (neighbor_idx_opt.has_value()) {
                        auto neighbor_idx = neighbor_idx_opt.value()->second;
                        std::vector<double> U_neighbor = {
                            m_tree.template get<amr::cell::Rho>(neighbor_idx),
                            m_tree.template get<amr::cell::Rhou>(neighbor_idx),
                            m_tree.template get<amr::cell::Rhov>(neighbor_idx),
                            m_tree.template get<amr::cell::E>(neighbor_idx)
                        };
                        // Sum up neighbor states for averaging
                        for (size_t k = 0; k < 4; ++k) {
                            U_neighbor_avg[k] += U_neighbor[k];
                        }
                    }
                }
                // Average the neighbor states and compute flux
                for (size_t k = 0; k < 4; ++k) {
                    U_neighbor_avg[k] /= neighbors_right.value().size();
                }
                m_euler_solver.rusanovFlux(U_current, U_neighbor_avg, flux_right, Direction::X);
            } else {
                // Boundary condition: No flux across the boundary
                // TODO: set the flux to 0, or reflect the state for a wall boundary
                // For now, let's assume zero gradient (neumann) so no action is needed here
            }

            // --- Calculate fluxes for the left face (X-direction) ---
            auto neighbors_left = m_tree.get_neighbors(node_id, TreeT::node_index_directon_t::left);
            if (neighbors_left.has_value()) {
                std::vector<double> U_neighbor_avg(4, 0.0);
                for (const auto& neighbor_id : neighbors_left.value()) {
                    auto neighbor_idx_opt = m_tree.find_index(neighbor_id);
                    if (neighbor_idx_opt.has_value()) {
                        auto neighbor_idx = neighbor_idx_opt.value()->second;
                        std::vector<double> U_neighbor = {
                            m_tree.template get<amr::cell::Rho>(neighbor_idx),
                            m_tree.template get<amr::cell::Rhou>(neighbor_idx),
                            m_tree.template get<amr::cell::Rhov>(neighbor_idx),
                            m_tree.template get<amr::cell::E>(neighbor_idx)
                        };
                        for (size_t k = 0; k < 4; ++k) {
                            U_neighbor_avg[k] += U_neighbor[k];
                        }
                    }
                }
                for (size_t k = 0; k < 4; ++k) {
                    U_neighbor_avg[k] /= neighbors_left.value().size();
                }
                m_euler_solver.rusanovFlux(U_neighbor_avg, U_current, flux_left, Direction::X);
            }

            // --- Calculate fluxes for the top face (Y-direction) ---
            auto neighbors_top = m_tree.get_neighbors(node_id, TreeT::node_index_directon_t::top);
            if (neighbors_top.has_value()) {
                std::vector<double> U_neighbor_avg(4, 0.0);
                for (const auto& neighbor_id : neighbors_top.value()) {
                    auto neighbor_idx_opt = m_tree.find_index(neighbor_id);
                    if (neighbor_idx_opt.has_value()) {
                        auto neighbor_idx = neighbor_idx_opt.value()->second;
                        std::vector<double> U_neighbor = {
                            m_tree.template get<amr::cell::Rho>(neighbor_idx),
                            m_tree.template get<amr::cell::Rhou>(neighbor_idx),
                            m_tree.template get<amr::cell::Rhov>(neighbor_idx),
                            m_tree.template get<amr::cell::E>(neighbor_idx)
                        };
                        for (size_t k = 0; k < 4; ++k) {
                            U_neighbor_avg[k] += U_neighbor[k];
                        }
                    }
                }
                for (size_t k = 0; k < 4; ++k) {
                    U_neighbor_avg[k] /= neighbors_top.value().size();
                }
                m_euler_solver.rusanovFlux(U_current, U_neighbor_avg, flux_top, Direction::Y);
            }

            // --- Calculate fluxes for the bottom face (Y-direction) ---
            auto neighbors_bottom = m_tree.get_neighbors(node_id, TreeT::node_index_directon_t::bottom);
            if (neighbors_bottom.has_value()) {
                std::vector<double> U_neighbor_avg(4, 0.0);
                for (const auto& neighbor_id : neighbors_bottom.value()) {
                    auto neighbor_idx_opt = m_tree.find_index(neighbor_id);
                    if (neighbor_idx_opt.has_value()) {
                        auto neighbor_idx = neighbor_idx_opt.value()->second;
                        std::vector<double> U_neighbor = {
                            m_tree.template get<amr::cell::Rho>(neighbor_idx),
                            m_tree.template get<amr::cell::Rhou>(neighbor_idx),
                            m_tree.template get<amr::cell::Rhov>(neighbor_idx),
                            m_tree.template get<amr::cell::E>(neighbor_idx)
                        };
                        for (size_t k = 0; k < 4; ++k) {
                            U_neighbor_avg[k] += U_neighbor[k];
                        }
                    }
                }
                for (size_t k = 0; k < 4; ++k) {
                    U_neighbor_avg[k] /= neighbors_bottom.value().size();
                }
                m_euler_solver.rusanovFlux(U_neighbor_avg, U_current, flux_bottom, Direction::Y);
            }
            
        
            
            // Apply the finite volume update to the temporary buffer
            double dx = 1.0 / (1 << node_id.level());
            double dy = 1.0 / (1 << node_id.level());
            
            for (size_t k = 0; k < 4; ++k) {
                U_new[i][k] = U_current[k] - (dt / dx) * (flux_right[k] - flux_left[k])
                                        - (dt / dy) * (flux_top[k] - flux_bottom[k]);
            }
        }

        // Scatter new state from the temporary buffer back to the tree
        for (size_t i = 0; i < m_tree.size(); ++i) {
            amr::cell::EulerCell new_cell(U_new[i][0], U_new[i][1], U_new[i][2], U_new[i][3]);
            m_tree.scatter_node(new_cell, i);
        }
    }
};

#endif // AMR_SOLVER_HPP