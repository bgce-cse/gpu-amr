#ifndef DG_HELPERS_SOLVER_HPP
#define DG_HELPERS_SOLVER_HPP

#include "containers/container_utils.hpp"
#include "containers/static_layout.hpp"
#include "containers/static_shape.hpp"
#include "containers/static_vector.hpp"
#include "dg_helpers/globals/amr_indicators.hpp"
#include "dg_helpers/globals/global_config.hpp"
#include "dg_helpers/rhs.hpp"
#include "dg_helpers/time_integration/time_integration.hpp"
#include "dg_helpers/tree_builder.hpp"
#include "generated_config.hpp"
#include "ndtree/print_dg_tree_v2.hpp"
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <string>

namespace amr::solver
{

/**
 * @brief Runtime parameters for the DG simulation loop.
 *
 * Separates knobs that might change between runs from the compile-time
 * policy so they can be loaded from a file or command-line later.
 */
struct SimulationParams
{
    double initial_dt    = 0.01;
    double plot_interval = 0.01;
    int    initial_level = 4;
    int    amr_interval  = 200;

    /// Gradient-based AMR thresholds
    amr::global::AMRThresholds amr_thresholds = { .refine_threshold  = 0.05,
                                                  .coarsen_threshold = 0.01 };
};

/**
 * @brief Discontinuous-Galerkin solver with tree-based AMR.
 *
 * @tparam Policy  Compile-time configuration policy
 *                 (e.g. amr::config::GlobalConfigPolicy).
 *
 * Owns the tree, time integrator, and VTK printer.
 * Call run() to execute the full time-stepping loop.
 */
template <typename Policy>
struct DGSolver
{
    // -----------------------------------------------------------------
    //  Type aliases – short names used throughout the solver
    // -----------------------------------------------------------------
    using global_t          = amr::global::GlobalConfig<Policy>;
    using rhs_t             = amr::rhs::RHSEvaluator<global_t, Policy>;
    using tree_builder_t    = amr::dg_tree::TreeBuilder<global_t, Policy>;
    using S1                = typename tree_builder_t::S1;
    using S2                = typename tree_builder_t::S2;
    using patch_index_t     = typename tree_builder_t::patch_index_t;
    using tree_t            = typename tree_builder_t::tree_t;
    using patch_layout_t    = typename tree_builder_t::patch_layout_t;
    using patch_container_t = amr::containers::
        static_tensor<typename S1::type, typename patch_layout_t::padded_layout_t>;
    using integrator_traits_t = amr::time_integration::
        TimeIntegratorTraits<Policy::integrator, patch_container_t>;
    using integrator_t    = typename integrator_traits_t::type;
    using printer_t       = ::ndt::print::dg_tree_printer_2d<global_t, Policy>;
    using amr_indicator_t = amr::global::GradientAMRIndicator<global_t, Policy>;
    using refine_status_t = typename tree_t::refine_status_t;

    // -----------------------------------------------------------------
    //  State
    // -----------------------------------------------------------------
    SimulationParams          params;
    tree_builder_t            tree_builder;
    integrator_t              integrator{};
    std::vector<integrator_t> patch_integrators;

    // -----------------------------------------------------------------
    //  Convenience accessors
    // -----------------------------------------------------------------
    auto& tree()
    {
        return tree_builder.tree;
    }

    const auto& tree() const
    {
        return tree_builder.tree;
    }

    // -----------------------------------------------------------------
    //  Construction
    // -----------------------------------------------------------------
    explicit DGSolver(SimulationParams p = {})
        : params{ p }
        , tree_builder{ p.initial_level }
        , patch_integrators(tree_builder.tree.size())
    {
    }

    // -----------------------------------------------------------------
    //  Public interface
    // -----------------------------------------------------------------

    /** Run the full simulation until Policy::EndTime. */
    void run()
    {
        printer_t printer("dg_tree");

        double time           = 0.0;
        double dt             = params.initial_dt;
        int    timestep       = 0;
        double next_plot_time = 0.01;
        int    next_amr_step  = 0;

        // tree_builder.initialize(4);
        write_output(printer, timestep, time);

        while (time < Policy::EndTime)
        {
            double dt_min = dt;
            advance_patches(dt, dt_min);
            dt = std::min(dt, dt_min);

            time += dt;
            ++timestep;

            apply_amr(timestep, next_amr_step);

            if (time >= next_plot_time) // time >= next_plot_time
            {
                write_output(printer, timestep, time);
                next_plot_time += params.plot_interval;
            }
        }
    }

private:
    // -----------------------------------------------------------------
    //  Time-step kernel: advance every patch by dt
    //
    //  Uses the stage-wise integrator interface so that a global halo
    //  exchange is performed between every RK stage.  This ensures that
    //  intermediate-stage RHS evaluations see consistent neighbor data
    //  instead of stale halos from the previous time level.
    // -----------------------------------------------------------------
    void advance_patches(double dt, double& dt_min)
    {
        auto&         t         = tree();
        constexpr int n_stages  = integrator_t::num_stages();
        const auto    n_patches = t.size();

        // Precompute CFL scale factor at compile time
        static constexpr double cfl_scale = []()
        {
            double r = 1.0;
            for (std::size_t i = 0; i < Policy::Dim; ++i)
                r *= Policy::Order;
            return 1.0 / (r + 1.0);
        }();

        // Resize only when tree size changes (after AMR)
        if (patch_integrators.size() != n_patches) patch_integrators.resize(n_patches);

        // --- Begin step: save u^n for every patch ---
        t.halo_exchange_update();
#pragma omp parallel for schedule(static)
        for (std::size_t idx = 0; idx < n_patches; ++idx)
        {
            auto& dof_patch = t.template get_patch<S1>(idx);
            patch_integrators[idx].begin_step(dof_patch.data());
        }

        // --- Stage loop ---
        for (int s = 0; s < n_stages; ++s)
        {
            // Halo exchange so every patch sees the consistent state
            // (u^n for stage 0, the intermediate u* for later stages).
            t.halo_exchange_update();

            double stage_dt_min = dt_min;

#pragma omp parallel for schedule(static) reduction(min : stage_dt_min)
            for (std::size_t idx = 0; idx < n_patches; ++idx)
            {
                auto& dof_patch  = t.template get_patch<S1>(idx);
                auto& flux_patch = t.template get_patch<S2>(idx);

                const double edge    = global_t::cell_edge(t.get_node_index_at(idx));
                const double volume  = global_t::cell_volume(edge);
                const double surface = global_t::cell_area(edge);
                const double inv_jac = 1.0 / edge;

                double max_eigenval = -std::numeric_limits<double>::infinity();

                auto residual = [&](patch_container_t&       update,
                                    const patch_container_t& current,
                                    double /*t*/)
                {
                    rhs_t::evaluate(
                        current,
                        flux_patch.data(),
                        update,
                        dt,
                        volume,
                        surface,
                        max_eigenval,
                        inv_jac
                    );
                };

                // Evaluate this stage's RHS using the halo-consistent tree data
                patch_integrators[idx].compute_stage(
                    s, residual, dof_patch.data(), 0.0, dt
                );

                // Write the intermediate result back into the tree so the
                // next halo exchange reflects it.
                dof_patch.data() =
                    patch_integrators[idx].stage_result(s, dof_patch.data());

                // CFL bookkeeping (only on the last stage to avoid redundant work)
                if (s == n_stages - 1 && max_eigenval > 0.0)
                {
                    const double new_dt =
                        cfl_scale * amr::config::CourantNumber * edge / max_eigenval;
                    stage_dt_min = std::min(stage_dt_min, new_dt);
                }
            }

            dt_min = stage_dt_min;
        }

// --- Finish step: write final u^{n+1} into tree ---
#pragma omp parallel for schedule(static)
        for (std::size_t idx = 0; idx < n_patches; ++idx)
        {
            auto& dof_patch = t.template get_patch<S1>(idx);
            patch_integrators[idx].finish_step(dof_patch.data());
        }
    }

    // -----------------------------------------------------------------
    //  AMR: conditionally refine / coarsen the tree
    // -----------------------------------------------------------------
    void apply_amr(int timestep, int& next_amr_step)
    {
        if (timestep < next_amr_step) return;

        // --- Gradient-based AMR with majority voting ---
        //
        // 1. Compute per-cell gradient indicators for every patch.
        // 2. Each interior cell votes Refine / Coarsen / Stable.
        // 3. Majority vote determines the patch-level decision.
        // 4. The resulting decision vector is indexed by the natural
        //    sequential patch layout [0 .. tree.size()-1].
        //
        // The tree's reconstruct_tree() already enforces
        // max-depth limits and 2:1 balance constraints, so we
        // just hand it the raw per-patch decision.
        auto decisions = amr_indicator_t::template compute_patch_decisions<tree_t, S1>(
            tree(), params.amr_thresholds
        );

        // --- Log AMR statistics ---
        int n_refine = 0, n_coarsen = 0, n_stable = 0;
        for (const auto& d : decisions)
        {
            if (d == refine_status_t::Refine)
                ++n_refine;
            else if (d == refine_status_t::Coarsen)
                ++n_coarsen;
            else
                ++n_stable;
        }
        const auto old_size = tree().size();
        std::cout << "[AMR] step=" << timestep << "  patches=" << old_size
                  << "  votes: refine=" << n_refine << " coarsen=" << n_coarsen
                  << " stable=" << n_stable << "\n";

        // Build a look-up map keyed by patch_index_t for the
        // tree's callback.
        std::unordered_map<patch_index_t, refine_status_t> decision_map;
        decision_map.reserve(decisions.size());
        for (std::size_t i = 0; i < decisions.size(); ++i)
            decision_map[tree().get_node_index_at(i)] = decisions[i];

        tree().reconstruct_tree(
            [&decision_map](const patch_index_t& pid) -> refine_status_t
            {
                auto it = decision_map.find(pid);
                if (it != decision_map.end()) return it->second;
                return refine_status_t::Stable;
            }
        );

        std::cout << "[AMR] after reconstruct: patches=" << tree().size() << "\n";

        // Halo exchange after AMR so newly created patches have valid halos
        tree().halo_exchange_update();

        next_amr_step = timestep + params.amr_interval;
    }

    // -----------------------------------------------------------------
    //  VTK output
    // -----------------------------------------------------------------
    void write_output(printer_t& printer, int timestep, double time)
    {
        const std::string ext = "_t" + std::to_string(timestep) + ".vtk";
        printer.template print<S1>(tree(), ext);
        std::cout << "Output " << ext << " time=" << time << "\n";
    }
};

} // namespace amr::solver

#endif // DG_HELPERS_SOLVER_HPP