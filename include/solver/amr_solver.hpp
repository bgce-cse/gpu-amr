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

            // Fetch the tuple of SoA patch references (same as time_step)
            auto fetch_patches = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                return std::forward_as_tuple(
                    m_tree.template get_patch<
                        typename std::tuple_element<Is, typename EquationT::FieldTags>::type
                    >(patch_id)...
                );
            };
            auto patches = fetch_patches(std::make_index_sequence<NVAR>{});

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
                
                // Write directly to the SoA arrays
                auto write_state = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                    ((std::get<Is>(patches)[l_idx] = cons[Is]), ...);
                };
                write_state(std::make_index_sequence<NVAR>{});
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

            // TODO: update_buffer will be replaced by ndtree double buffering of m_data_buffers
            std::array<std::vector<double>, NVAR> update_buffer;
            for (int k = 0; k < NVAR; ++k) update_buffer[k].resize(patch_flat_size, 0.0);

            // Extract a tuple of references to all the SoA patches for this node AT ONCE
            auto fetch_patches = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                return std::forward_as_tuple(
                    m_tree.template get_patch<
                        typename std::tuple_element<Is, typename EquationT::FieldTags>::type
                    >(patch_id)...
                );
            };
            auto patches = fetch_patches(std::make_index_sequence<NVAR>{});

            for (std::size_t l_idx = 0; l_idx < patch_flat_size; ++l_idx) {
                if (amr::ndt::utils::patches::is_halo_cell<PatchLayoutT>(l_idx)) continue;

                amr::containers::static_vector<double, NVAR> total_update = amr::containers::static_vector<double, NVAR>{};

                // Generic loop over dimensions (X, Y, Z)
                for (int d = 0; d < DIM; ++d) {
                    size_t stride = (d == 0) ? 1 : (d == 1 ? stride_y : stride_z);

                    amr::containers::static_vector<double, NVAR> fL, fR;
                    EquationT::rusanovFlux(patches, l_idx-stride, l_idx, fL, d, gamma);
                    EquationT::rusanovFlux(patches, l_idx, l_idx+stride, fR, d, gamma);

                    for (int k = 0; k < NVAR; ++k) {
                        total_update[k] -= (dt / c_size[d]) * (fR[k] - fL[k]);
                    }
                }

                auto apply_update = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                    ((update_buffer[Is][l_idx] = std::get<Is>(patches)[l_idx] + total_update[Is]), ...);
                };
                apply_update(std::make_index_sequence<NVAR>{});
            }

            auto write_back = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                for (std::size_t l_idx = 0; l_idx < patch_flat_size; ++l_idx) {
                    if (!amr::ndt::utils::patches::is_halo_cell<PatchLayoutT>(l_idx)) {
                        // std::get<Is>(patches) gets the patch, [l_idx] accesses the element
                        ((std::get<Is>(patches)[l_idx] = update_buffer[Is][l_idx]), ...);
                    }
                }
            };
            write_back(std::make_index_sequence<NVAR>{});
        }
    }

    double compute_time_step() const {
        double dt_min = std::numeric_limits<double>::max();

        for (std::size_t p_idx = 0; p_idx < m_tree.size(); ++p_idx) {
            auto patch_id = m_tree.get_node_index_at(p_idx);
            auto c_size = GeometryT::cell_sizes(patch_id);

            // Fetch the SoA patch tuple outside the cell loop
            auto fetch_patches = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                return std::forward_as_tuple(
                    m_tree.template get_patch<
                        typename std::tuple_element<Is, typename EquationT::FieldTags>::type
                    >(patch_id)...
                );
            };
            auto patches = fetch_patches(std::make_index_sequence<NVAR>{});

            for (std::size_t l_idx = 0; l_idx < PatchLayoutT::flat_size(); ++l_idx) {
                if (amr::ndt::utils::patches::is_halo_cell<PatchLayoutT>(l_idx)) continue;

                // Ask Equation for max wave speed directly from the SoA tuple
                for (int d = 0; d < DIM; ++d) {
                    double speed = EquationT::getMaxSpeedSoA(patches, l_idx, d, gamma);
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