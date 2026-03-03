#ifndef AMR_SOLVER_HPP
#define AMR_SOLVER_HPP

#include "EulerPhysics.hpp"
#include "cell_types.hpp"
#include "config/definitions.hpp"
#include "containers/container_manipulations.hpp"
#include "ndtree/ndtree.hpp"
#include "physics_system.hpp"
#include <algorithm>
#include <execution>
#include <limits>
#include <vector>

/**
 * @brief AMR Solver
 * @tparam TreeT The ndtree structure
 * @tparam GeometryT The physics_system (coordinates/sizes)
 * @tparam EquationT The physical model (Euler, Advection, etc.)
 * @tparam DIM Dimensionality
 */
template <typename TreeT, typename GeometryT, typename EquationT, int DIM>
class amr_solver
{
public:
    using tree_t               = TreeT;
    using patch_layout_t       = typename tree_t::patch_layout_t;
    using patch_index_t        = typename tree_t::patch_index_t;
    using linear_index_t       = typename tree_t::linear_index_t;
    using data_layout_t        = typename patch_layout_t::data_layout_t;
    using padded_layout_t      = typename patch_layout_t::padded_layout_t;
    static constexpr auto NVAR = EquationT::NVAR;
    // TODO: Make a template
    using arithmetic_t = double;

private:
    TreeT              m_tree;
    arithmetic_t const m_gamma; // Specific heat ratio
    arithmetic_t const m_cfl;   // CFL number

public:
    amr_solver(size_t capacity, arithmetic_t gamma_ = 1.4, arithmetic_t cfl_ = 0.1)
        : m_tree(capacity)
        , m_gamma(gamma_)
        , m_cfl(cfl_)
    {
        // Dummy dimensions were removed from new EulerPhysics.hpp
        static_assert(DIM == 2 || DIM == 3, "Error: Wrong dimension");
    }

    TreeT& get_tree()
    {
        return m_tree;
    }

    /**
     * @brief Helper to gather the full conservative state from a specific cell
     */
    auto
        get_full_state(const linear_index_t patch_idx, const std::size_t linear_idx) const
    {
        // Static loop over the FieldTags defined in the Equation policy
        auto fill_state = [&]<std::size_t... Is>(std::index_sequence<Is...>)
        {
            amr::containers::static_vector<arithmetic_t, NVAR> state;
            ((state[Is] = m_tree.template get_patch<
                          typename std::tuple_element<Is, typename EquationT::FieldTags>::
                              type>(patch_idx)[linear_idx]),
             ...);
            return state;
        };
        return fill_state(std::make_index_sequence<NVAR>{});
    }

    void set_full_state(
        const linear_index_t                                      patch_idx,
        const std::size_t                                         linear_idx,
        const amr::containers::static_vector<arithmetic_t, NVAR>& state
    )
    {
        auto write_state = [&]<std::size_t... Is>(std::index_sequence<Is...>) -> void
        {
            ((m_tree.template get_patch<
                  typename std::tuple_element<Is, typename EquationT::FieldTags>::type>(
                  patch_idx
              )[linear_idx] = state[Is]),
             ...);
        };
        write_state(std::make_index_sequence<NVAR>{});
    }

    arithmetic_t get_gamma() const noexcept
    {
        return m_gamma;
    }

    arithmetic_t get_cfl() const noexcept
    {
        return m_cfl;
    }

    void initialize(auto&& init_func)
    {
        for (std::size_t p_idx = 0; p_idx < m_tree.size(); ++p_idx)
        {
            const auto patch_id = m_tree.get_node_index_at(p_idx);
            const auto c_size   = GeometryT::cell_sizes(patch_id);

            amr::containers::manipulators::shaped_for<
                typename patch_layout_t::interior_iteration_control_t>(
                [this, p_idx, patch_id, &c_size](auto&& fn, auto const& idxs)
                {
                    const auto linear_idx = padded_layout_t::linear_index(idxs);
                    // Dimension-agnostic coordinate fetch
                    const auto cell_origin = GeometryT::cell_coord(patch_id, linear_idx);
                    std::array<arithmetic_t, DIM> coords;
                    for (int d = 0; d < DIM; ++d)
                        coords[d] = cell_origin[d] + 0.5 * c_size[d];

                    // IC -> Primitive -> Conservative
                    const auto prim = std::invoke(std::forward<decltype(fn)>(fn), coords);
                    amr::containers::static_vector<arithmetic_t, NVAR> cons;
                    EquationT::primitiveToConservative(prim, cons, m_gamma);

                    set_full_state(p_idx, linear_idx, cons);
                },
                std::forward<decltype(init_func)>(init_func)
            );
        }
    }

    auto time_step(const arithmetic_t dt) -> void
    {
        constexpr auto patch_data_size = data_layout_t::flat_size();
        // Stride for moving "up" or "down" in the patch (Y-direction)
        constexpr auto stride_y =
            patch_layout_t::padded_layout_t::shape_t::sizes()[DIM - 1];
        // Stride for Z (only used if DIM=3)
        // TODO: Maybe use the stride accessors layout types provide
        constexpr auto stride_z =
            (DIM == 3) ? patch_layout_t::padded_layout_t::shape_t::sizes()[1] * stride_y
                       : 0;

        for (std::size_t p_idx = 0; p_idx < m_tree.size(); ++p_idx)
        {
            const auto patch_id = m_tree.get_node_index_at(p_idx);
            const auto c_size   = GeometryT::cell_sizes(patch_id);

            std::
                array<amr::containers::static_vector<arithmetic_t, NVAR>, patch_data_size>
                    update_buffer{};

            int buffer_idx = 0;
            amr::containers::manipulators::shaped_for<
                typename patch_layout_t::interior_iteration_control_t>(
                [this, p_idx, dt, &c_size](
                    auto& out_buffer, auto& out_i, auto const& idxs
                )
                {
                    const auto linear_idx = padded_layout_t::linear_index(idxs);
                    const auto U_cell     = get_full_state(p_idx, linear_idx);
                    out_buffer[out_i]     = U_cell;

                    // Generic loop over dimensions (X, Y, Z)
                    for (int d = 0; d < DIM; ++d)
                    {
                        const auto stride = (d == 0) ? 1 : (d == 1 ? stride_y : stride_z);

                        const auto U_L = get_full_state(p_idx, linear_idx - stride);
                        const auto U_R = get_full_state(p_idx, linear_idx + stride);

                        // TODO: Remove out paramters if possible
                        // TODO: Evaluate merging both calls into one since they share
                        // input params
                        amr::containers::static_vector<arithmetic_t, NVAR> fL, fR;
                        EquationT::rusanovFlux(U_L, U_cell, fL, d, m_gamma);
                        EquationT::rusanovFlux(U_cell, U_R, fR, d, m_gamma);

                        for (int k = 0; k < NVAR; ++k)
                        {
                            out_buffer[out_i][k] -= (dt / c_size[d]) * (fR[k] - fL[k]);
                        }
                    }
                    out_i++;
                },
                update_buffer,
                buffer_idx
            );
            buffer_idx = 0;
            amr::containers::manipulators::shaped_for<
                typename patch_layout_t::interior_iteration_control_t>(
                [this, p_idx, &update_buffer](auto& out_i, auto const& idxs)
                {
                    const auto linear_idx = padded_layout_t::linear_index(idxs);
                    set_full_state(p_idx, linear_idx, update_buffer[out_i++]);
                },
                buffer_idx
            );
        }
    }

    auto compute_time_step() const -> arithmetic_t
    {
        // TODO: We could have a tighter upper bound here.
        //       maybe there is some other theoretical limit we can use instead
        std::atomic<arithmetic_t> dt{ std::numeric_limits<arithmetic_t>::max() };

        auto const r = std::views::iota(decltype(m_tree.size()){}, m_tree.size());
        std::for_each(
            AMR_EXECUTION_POLICY,
            std::cbegin(r),
            std::cend(r),
            [this, &dt](auto const p_idx) mutable
            {
                auto       local_dt{ std::numeric_limits<arithmetic_t>::max() };
                const auto patch_id = m_tree.get_node_index_at(p_idx);
                const auto c_size   = GeometryT::cell_sizes(patch_id);

                amr::containers::manipulators::shaped_for<
                    typename patch_layout_t::interior_iteration_control_t>(
                    [this, p_idx, &c_size](arithmetic_t& out_dt, auto const& idxs)
                    {
                        const auto linear_idx = padded_layout_t::linear_index(idxs);
                        const auto U          = get_full_state(p_idx, linear_idx);

                        // Ask Equation for max wave speed in each direction
                        // TODO: The level of abstraction here is incorrect in my
                        // opiniton. Iterating over the dimensions and calling it
                        // direction is heavily missleading.
                        // Look at neighbor::direction as a suggested alternative.
                        for (int d = 0; d < DIM; ++d)
                        {
                            const arithmetic_t speed =
                                EquationT::getMaxSpeed(U, d, m_gamma);
                            // TODO: This magic number should at least have a name
                            // TODO: Is this value special in any way?
                            if (speed > 1e-12)
                            {
                                out_dt = std::min(out_dt, c_size[d] / speed);
                            }
                        }
                    },
                    local_dt
                );
                auto current = dt.load();
                while (local_dt < current && !dt.compare_exchange_weak(current, local_dt))
                {
                    break;
                }
            }
        );
        return m_cfl * dt;
    }
};

// Typedefs
template <typename TreeT, typename GeometryT, typename EquationT>
using amr_solver_2d = amr_solver<TreeT, GeometryT, EquationT, 2>;

template <typename TreeT, typename GeometryT, typename EquationT>
using amr_solver_3d = amr_solver<TreeT, GeometryT, EquationT, 3>;

#endif // AMR_SOLVER_HPP
