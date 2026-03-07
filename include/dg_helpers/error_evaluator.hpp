#ifndef DG_HELPERS_ERROR_EVALUATOR_HPP
#define DG_HELPERS_ERROR_EVALUATOR_HPP

#include "basis/basis.hpp"
#include "globals/coordinates.hpp"
#include "globals/global_config.hpp"
#include "ndtree/patch_utils.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <iostream>

namespace amr::error
{

/**
 * @brief L1, L2, Linf error norms per DOF variable.
 */
template <std::size_t NumDOF>
struct ErrorNorms
{
    std::array<double, NumDOF> l1{};
    std::array<double, NumDOF> l2{};
    std::array<double, NumDOF> linf{};
};

/**
 * @brief Evaluate integral error norms of numerical solution vs. analytical.
 *
 * For each leaf patch / interior cell, uses a higher-order quadrature
 * (max(Order+2, 6)) to integrate the pointwise error. The numerical
 * solution is evaluated via basis interpolation, the analytical solution
 * via the equation's get_initial_values at the current time.
 *
 * @tparam global_t  Fully-assembled GlobalConfig type.
 * @tparam Policy    Compile-time configuration policy.
 */
template <typename global_t, typename Policy>
struct ErrorEvaluator
{
    static constexpr std::size_t Dim   = Policy::Dim;
    static constexpr std::size_t Order = Policy::Order;
    static constexpr std::size_t DOFs  = Policy::DOFs;

    using eq_t  = typename global_t::EquationImpl;
    using Basis = typename global_t::Basis;

    // Higher-order quadrature for accurate error integration.
    // Use max(Order+2, 6) but cap at 8 (our max QuadData specialization).
    static constexpr std::size_t QuadOrder =
        (Order + 2 > 6) ? (Order + 2 > 8 ? 8 : Order + 2) : 6;

    using HighQuad     = amr::basis::GaussLegendre<QuadOrder, 0.0, 1.0>;
    using HighLagrange = amr::basis::Lagrange<Order>;
    using coord_vector = amr::containers::static_vector<double, Dim>;

    /**
     * @brief Evaluate error norms over the entire tree.
     *
     * @tparam TreeT   ndtree type
     * @tparam S1Tag   DOF patch tag
     * @param tree     The tree holding leaf patches
     * @param time     Current simulation time (for analytical solution)
     * @return ErrorNorms<DOFs>  The L1, L2, Linf norms per variable.
     */
    template <typename TreeT, typename S1Tag>
    static auto evaluate(const TreeT& tree, double time) -> ErrorNorms<DOFs>
    {
        using patch_layout_t = typename TreeT::patch_layout_t;

        ErrorNorms<DOFs> norms{};

        for (std::size_t p = 0; p < tree.size(); ++p)
        {
            const auto& dof_patch = tree.template get_patch<S1Tag>(p);
            auto        patch_id  = tree.get_node_index_at(p);

            const double cell_edge = global_t::cell_edge(patch_id);
            const double cell_vol  = global_t::cell_volume(cell_edge);

            for (std::size_t l_idx = 0; l_idx < patch_layout_t::flat_size(); ++l_idx)
            {
                if (amr::ndt::utils::patches::is_halo_cell<patch_layout_t>(l_idx))
                    continue;

                // Get cell center in global coordinates
                auto coords_with_halo = global_t::lin_to_local(l_idx);
                auto local_indices    = global_t::rm_halo(coords_with_halo);
                auto cell_center      = global_t::compute_center(patch_id, local_indices);

                const auto& cell_dof = dof_patch[l_idx];

                // Integrate error with higher-order quadrature
                integrate_cell_error(
                    cell_dof, cell_center, cell_edge, time, cell_vol, norms
                );
            }
        }

        // Finalize L2 norm
        for (std::size_t v = 0; v < DOFs; ++v)
            norms.l2[v] = std::sqrt(norms.l2[v]);

        return norms;
    }

    /**
     * @brief Print error norms in a readable table.
     */
    static void print(const ErrorNorms<DOFs>& norms)
    {
        std::cout << "\nError norms per variable:\n";
        std::cout << "Var\t     L1\t\t\t     L2\t\t\t     Linf\n";
        std::cout << "---\t---------------\t\t---------------\t\t---------------\n";
        for (std::size_t v = 0; v < DOFs; ++v)
        {
            std::printf(
                "%zu\t%15.6e\t\t%15.6e\t\t%15.6e\n",
                v,
                norms.l1[v],
                norms.l2[v],
                norms.linf[v]
            );
        }
        std::cout << "\n";
    }

private:
    /**
     * @brief Integrate pointwise error over one DG cell using the
     *        higher-order quadrature rule.
     *
     * For each high-order quadrature point:
     *   1. Map it to reference coords of the DG basis ([0,1]^Dim)
     *   2. Evaluate the numerical solution via Lagrange interpolation
     *   3. Map it to global coords for the analytical solution
     *   4. Accumulate weighted error into L1, L2, Linf
     */
    template <typename CellDOF>
    static void integrate_cell_error(
        const CellDOF&      cell_dof,
        const coord_vector& cell_center,
        double              cell_edge,
        double              time,
        double              cell_vol,
        ErrorNorms<DOFs>&   norms
    )
    {
        // Gauss-Legendre points are already on [0,1].
        // The DG basis nodes are also on [0,1], so we evaluate
        // the basis directly at the quadrature points.
        //
        // For global coordinates: x_global = cell_center + (xi - 0.5) * cell_edge
        // (reference [0,1] centered at 0.5)

        if constexpr (Dim == 2)
        {
            for (std::size_t i = 0; i < QuadOrder; ++i)
            {
                for (std::size_t j = 0; j < QuadOrder; ++j)
                {
                    const double qx = HighQuad::points[i];
                    const double qy = HighQuad::points[j];
                    const double qw = HighQuad::weights[i] * HighQuad::weights[j];

                    // Weight includes cell volume scaling
                    const double weight = cell_vol * qw;

                    // Evaluate numerical solution at this quadrature point
                    coord_vector ref_pt{};
                    ref_pt[0]      = qx;
                    ref_pt[1]      = qy;
                    auto numerical = Basis::evaluate_basis(cell_dof, ref_pt);

                    // Map to global coordinates for analytical solution
                    coord_vector global_pt{};
                    global_pt[0]    = cell_center[0] + (qx - 0.5) * cell_edge;
                    global_pt[1]    = cell_center[1] + (qy - 0.5) * cell_edge;
                    auto analytical = eq_t::get_initial_values(global_pt, time);

                    // Accumulate errors per variable
                    for (std::size_t v = 0; v < DOFs; ++v)
                    {
                        const double err = std::abs(numerical[v] - analytical[v]);
                        norms.l1[v] += weight * err;
                        norms.l2[v] += weight * err * err;
                        norms.linf[v] = std::max(norms.linf[v], err);
                    }
                }
            }
        }
        else if constexpr (Dim == 3)
        {
            for (std::size_t i = 0; i < QuadOrder; ++i)
            {
                for (std::size_t j = 0; j < QuadOrder; ++j)
                {
                    for (std::size_t k = 0; k < QuadOrder; ++k)
                    {
                        const double qx = HighQuad::points[i];
                        const double qy = HighQuad::points[j];
                        const double qz = HighQuad::points[k];
                        const double qw = HighQuad::weights[i] * HighQuad::weights[j] *
                                          HighQuad::weights[k];

                        const double weight = cell_vol * qw;

                        coord_vector ref_pt{};
                        ref_pt[0]      = qx;
                        ref_pt[1]      = qy;
                        ref_pt[2]      = qz;
                        auto numerical = Basis::evaluate_basis(cell_dof, ref_pt);

                        coord_vector global_pt{};
                        global_pt[0]    = cell_center[0] + (qx - 0.5) * cell_edge;
                        global_pt[1]    = cell_center[1] + (qy - 0.5) * cell_edge;
                        global_pt[2]    = cell_center[2] + (qz - 0.5) * cell_edge;
                        auto analytical = eq_t::get_initial_values(global_pt, time);

                        for (std::size_t v = 0; v < DOFs; ++v)
                        {
                            const double err = std::abs(numerical[v] - analytical[v]);
                            norms.l1[v] += weight * err;
                            norms.l2[v] += weight * err * err;
                            norms.linf[v] = std::max(norms.linf[v], err);
                        }
                    }
                }
            }
        }
        else if constexpr (Dim == 1)
        {
            for (std::size_t i = 0; i < QuadOrder; ++i)
            {
                const double qx     = HighQuad::points[i];
                const double weight = cell_vol * HighQuad::weights[i];

                coord_vector ref_pt{};
                ref_pt[0]      = qx;
                auto numerical = Basis::evaluate_basis(cell_dof, ref_pt);

                coord_vector global_pt{};
                global_pt[0]    = cell_center[0] + (qx - 0.5) * cell_edge;
                auto analytical = eq_t::get_initial_values(global_pt, time);

                for (std::size_t v = 0; v < DOFs; ++v)
                {
                    const double err = std::abs(numerical[v] - analytical[v]);
                    norms.l1[v] += weight * err;
                    norms.l2[v] += weight * err * err;
                    norms.linf[v] = std::max(norms.linf[v], err);
                }
            }
        }
    }
};

} // namespace amr::error

#endif // DG_HELPERS_ERROR_EVALUATOR_HPP
