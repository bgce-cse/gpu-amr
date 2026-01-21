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

public:
    using PatchLayoutT = typename TreeT::patch_layout_t;
    using PatchIndexT = typename TreeT::patch_index_t;
    static constexpr int NVAR = EquationT::NVAR;

    amr_solver(size_t capacity, double gamma_ = 1.4, double cfl_ = 0.1)
        : m_tree(capacity), gamma(gamma_), cfl(cfl_) {
        // Dummy dimensions were removed from new EulerPhysics.hpp
        static_assert(DIM == 2 || DIM == 3, "Error: Wrong dimensions");
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

#endif // AMR_SOLVER_HPP