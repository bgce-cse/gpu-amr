#include "../build/generated_config.hpp"
#include "dg_helpers/error_evaluator.hpp"
#include "dg_helpers/solver.hpp"
#include <exception>
#include <iostream>

// ============================================================================
//  Identical to dg_main_loop, but at the end compares the L2 norm of
//  DOF 0 against the analytical solution.
// ============================================================================

#ifndef ADVECTION_AMR_L2_TOLERANCE
#    define ADVECTION_AMR_L2_TOLERANCE 0.2
#endif

int main()
{
    std::cout << "====================================\n"
              << "  DG Solver - AMR + Error Check\n"
              << "====================================\n\n";

    try
    {
        using policy_t   = amr::config::GlobalConfigPolicy;
        using global_t   = amr::global::GlobalConfig<policy_t>;
        using error_eval = amr::error::ErrorEvaluator<global_t, policy_t>;
        using solver_t   = amr::solver::DGSolver<policy_t>;
        using S1         = typename solver_t::S1;
        using tree_t     = typename solver_t::tree_t;

        amr::solver::SimulationParams params{
            .initial_dt    = 0.01,
            .plot_interval = 0.01,
            .initial_level = 5,
            .amr_interval  = 20,
        };

        solver_t solver(params);
        solver.run();

        // Error evaluation at t = EndTime (variable 0 only)
        const double final_time = policy_t::EndTime;

        std::cout << "\n====================================\n"
                  << "  Error Evaluation at t = " << final_time << "\n"
                  << "====================================\n";

        auto norms = error_eval::template evaluate<tree_t, S1>(solver.tree(), final_time);
        error_eval::print(norms);

        constexpr double tolerance = ADVECTION_AMR_L2_TOLERANCE;
        const double     l2_var0   = norms.l2[0];

        std::cout << "L2 error (var 0): " << l2_var0 << "\n"
                  << "Tolerance:        " << tolerance << "\n";

        if (l2_var0 < tolerance)
        {
            std::cout << "\n*** TEST PASSED ***\n";
            return 0;
        }
        else
        {
            std::cout << "\n*** TEST FAILED ***\n"
                      << "L2 error " << l2_var0 << " exceeds tolerance " << tolerance
                      << "\n";
            return 1;
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }
}
