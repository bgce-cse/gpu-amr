#include "../build/generated_config.hpp"
#include "dg_helpers/solver.hpp"
#include <exception>
#include <iostream>

int main()
{
    std::cout << "====================================\n"
              << "  DG Solver - AMR Main Loop\n"
              << "====================================\n\n";

    try
    {
        using policy_t = amr::config::GlobalConfigPolicy;

        amr::solver::SimulationParams params{
            .initial_dt    = 0.01,
            .plot_interval = 0.01,
            .initial_level = 5,
            .amr_interval  = 20,
        };

        amr::solver::DGSolver<policy_t> solver(params);
        solver.run();
    }
    catch (const std::exception& e)
    {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
