#pragma once

#include <memory>
#include <string>
#include <vector>

#include "Discretization.hpp"
#include "Fields.hpp"
#include "PressureSolver.hpp"
#include "Utils.hpp"
#include "domain.hpp"
#include "print_tree.hpp"

/**
 * @brief Class to hold and orchestrate the simulation flow.
 *
 */
class Case
{
  public:
    /**
     * @brief Parallel constructor for the Case.
     *
     * Reads input file, creates Fields, Grid, Boundary, Solver and sets
     * Discretization parameters. Creates output directory.
     *
     * @param[in] Input file name
     */
    Case(
        std::string const& file_name, int itermax, double epsilon,
        double output_freq, double recosntruction_freq, double vort_lower,
        double vort_upper
    );

    /**
     * @brief Main function to simulate the flow until the end time.
     *
     * Calculates the fluxes
     * Calculates the right hand side
     * Solves pressure
     * Calculates velocities
     * Outputs the solution files
     */
    void simulate(
        sim_domain& domain, Fields& field, SOR const& pressure_solver,
        const double t_end
    );

  private:
    /// Plain case name without paths
    std::string _case_name;
    /// Output directory name
    std::string _dir_name;
    /// Geometry file name
    std::string _geom_name{"NONE"};
    /// Relative input file path
    std::string _prefix;

    /// Solution file outputting frequency
    double _output_freq;

    // frequency of adapting the tree
    double _reconstruction_freq;

    double _vorticity_lower_limit;
    double _vorticity_upper_limit;

    vtk_printer _printer;

    /// Solver convergence tolerance
    double _tolerance;

    /// Maximum number of iterations for the solver
    int _max_iter;

    /// energy equation flag
    bool _energy_eq;

    /**
     * @brief Creating file names from given input data file
     *
     * Extracts path of the case file and creates code-readable file names
     * for outputting directory and geometry file.
     *
     * @param[in] input data file
     */
    void set_file_names(std::string const& file_name);
};
