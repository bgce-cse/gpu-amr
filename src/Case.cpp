#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <vector>

namespace filesystem = std::filesystem;

#include <cassert>
#include <vtkCellData.h>
#include <vtkDoubleArray.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkSmartPointer.h>
#include <vtkStructuredGrid.h>
#include <vtkStructuredGridWriter.h>
#include <vtkTuple.h>

#include "Case.hpp"
#include "PressureSolver.hpp"
#include "Utils.hpp"

Case::Case(
    std::string const& file_name, int itermax, double epsilon,
    double output_freq, double reconstruction_freq, double vort_lower,
    double vort_upper
)
{
    set_file_names(file_name);

    _printer = vtk_printer(_dir_name);

    _max_iter = itermax;
    _tolerance = epsilon;
    _output_freq = output_freq;
    _reconstruction_freq = reconstruction_freq;
    _vorticity_lower_limit = vort_lower;
    _vorticity_upper_limit = vort_upper;
}

void Case::set_file_names(std::string const& file_name)
{
    std::string temp_dir;
    bool case_name_flag = true;
    bool prefix_flag = false;

    for (auto i = (int)file_name.size() - 1; i > -1; --i)
    {
        if (file_name[i] == '/')
        {
            case_name_flag = false;
            prefix_flag = true;
        }
        if (case_name_flag)
        {
            _case_name.push_back(file_name[i]);
        }
        if (prefix_flag)
        {
            _prefix.push_back(file_name[i]);
        }
    }

    for (int i = (int)file_name.size() - (int)_case_name.size() - 1; i > -1;
         --i)
    {
        temp_dir.push_back(file_name[i]);
    }

    std::reverse(_case_name.begin(), _case_name.end());
    std::reverse(_prefix.begin(), _prefix.end());
    std::reverse(temp_dir.begin(), temp_dir.end());

    _case_name.erase(_case_name.size() - 4);
    _dir_name = temp_dir;
    _dir_name.append(_case_name);
    _dir_name.append("_Output");

    if (_geom_name.compare("NONE") != 0)
    {
        _geom_name = _prefix + _geom_name;
    }

    // Create output directory
    filesystem::path folder(_dir_name);
    try
    {
        filesystem::create_directory(folder);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Output directory could not be created." << std::endl;
        std::cerr << "Make sure that you have write permissions to the "
                     "corresponding location"
                  << std::endl;
    }
}

void Case::simulate(
    sim_domain& domain, Fields& field, SOR const& pressure_solver,
    const double t_end
)
{
    using tree_t = typename sim_domain::tree_t;
    int reconstruction_counter = 1;
    double t = 0.0;
    int timestep = 0;
    int output_counter = 1; // set to 1 bc of initial output
    double progress_interval = 0.002;
    double show_progress_time = t;
    std::size_t total_sor_iterations{};
    std::uint32_t sor_checksum{};
    _printer.print(domain, timestep);

    // vorticity refinement condition
    auto ref_condition = [&domain, this](const index_t& morton_idx) {
        if (domain.tree().get_cell_type(morton_idx) == cell_type::OBSTACLE)
        {
            return tree_t::refine_status_t::Coarsen;
        }

        const auto [_, level] = index_t::decode(morton_idx);

        auto f_top = domain.get_neighbor_value<f_t>(morton_idx, direction_t::N);
        auto f_bottom =
            domain.get_neighbor_value<f_t>(morton_idx, direction_t::S);
        auto g_right =
            domain.get_neighbor_value<g_t>(morton_idx, direction_t::E);
        auto g_left =
            domain.get_neighbor_value<g_t>(morton_idx, direction_t::W);

        auto dx = domain.dx(morton_idx);
        auto dy = domain.dy(morton_idx);

        double vort = (g_right - g_left) / dx - (f_top - f_bottom) / dy;
        double vort_mag = std::abs(vort);

        if (vort_mag > this->_vorticity_upper_limit &&
            level < index_t::max_depth() - 1)
        {
            return tree_t::refine_status_t::Refine;
        }
        if (vort_mag < this->_vorticity_lower_limit)
        {
            return tree_t::refine_status_t::Coarsen;
        }
        return tree_t::refine_status_t::Stable;
    };

    std::chrono::steady_clock::time_point start =
        std::chrono::steady_clock::now();

    bool safety_flag = true;
    const auto safety_check = [&safety_flag](auto const dt) {
        return safety_flag = dt > 1e-10;
    };
    utils::loop_progress lp(t, t_end);
    for (; t < t_end;)
    {
        const double dt = field.calculate_dt(domain);

        if (!safety_check(dt))
        {
            std::cout << "Error: Time step too small\n";
            break;
        }

        if (field.energy_eq())
        {
            field.calculate_temperatures(domain);
        }

        field.calculate_fluxes(domain);

        field.calculate_rs(domain);
        double res{};
        auto sor_iter = decltype(_max_iter){};

        for (; sor_iter != _max_iter;)
        {
            res = pressure_solver.solve(domain);
            // Increment the counter out of the loop control to also count the
            // last iteration when early exiting the loop
            ++sor_iter;
            if (res < _tolerance)
            {
                break;
            }
        }
        total_sor_iterations += static_cast<std::uint32_t>(sor_iter);
        sor_checksum =
            (sor_checksum << 1) ^ static_cast<std::uint32_t>(sor_iter);

        field.calculate_velocities(domain);

        lp.increment(field.dt());
        t += field.dt();

        if (show_progress_time < (t / t_end) || t >= t_end)
        {
            show_progress_time += progress_interval;
            lp.print_progress('\t');
            std::cout << " t: " << std::fixed << std::setprecision(5)
                      << std::setw(8) << t << "\tdt: " << std::setw(8) << dt
                      << "\tfinal residual: " << std::setw(8)
                      << std::setprecision(6) << res
                      << "\tSOR iterations: " << sor_iter << '\n';
        }
        if (t > output_counter * _output_freq || t >= t_end)
        {
            _printer.print(domain, timestep);
            output_counter += 1;
        }
        if (t > reconstruction_counter * _reconstruction_freq)
        {
            std::cout << "reconstruction for the " << reconstruction_counter
                      << " time" << std::endl;
            domain.tree().reconstruct_tree(ref_condition);
            _printer.print(domain, timestep);
            reconstruction_counter += 1;
        }
        timestep++;
    }

    const auto stop = std::chrono::steady_clock::now();

    if (safety_flag)
    {
        std::cout << "INFO: Run complete\n";
        const auto duration =
            std::chrono::duration_cast<std::chrono::duration<double>>(
                stop - start
            )
                .count();
        std::cout << "Simulation completed in " << duration << " seconds\n"
                  << "Total SOR iterations:\t" << total_sor_iterations
                  << "\nSOR Checksum:\t\t" << sor_checksum << '\n';
    }
    else
    {
        std::cout << "INFO: Run stopped for safety.\n";
        std::cout << "Timestep too small to ensure stability.\n";
    }
}
