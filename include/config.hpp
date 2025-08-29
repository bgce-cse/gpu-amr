#pragma once

#include "data_types.hpp"
#include "utility/error_handling.hpp"
#include <cassert>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

struct sim_config
{
    static auto parse_config(
        std::string const& file_name, [[maybe_unused]] int argn,
        [[maybe_unused]] char** args
    )
    {
        const int MAX_LINE_LENGTH = 1024;
        sim_config c;
        std::ifstream file(file_name);
        if (file.is_open())
        {
            std::string var;
            while (!file.eof() && file.good())
            {
                var = "";
                file >> var;
                if (var == "")
                {
                    continue;
                }
                else if (var[0] == '#')
                { /* ignore comment line*/
                    file.ignore(MAX_LINE_LENGTH, '\n');
                }
                else
                {
                    if (var == "xlength")
                    {
                        file >> c.xlength;
                    }
                    else if (var == "ylength")
                    {
                        file >> c.ylength;
                    }
                    else if (var == "nu")
                    {
                        file >> c.nu;
                    }
                    else if (var == "alpha")
                    {
                        file >> c.alpha;
                    }
                    else if (var == "beta")
                    {
                        file >> c.beta;
                    }
                    else if (var == "TI")
                    {
                        file >> c.TI;
                    }
                    else if (var == "t_end")
                    {
                        file >> c.t_end;
                    }
                    else if (var == "dt")
                    {
                        file >> c.dt;
                    }
                    else if (var == "omg")
                    {
                        file >> c.omega;
                    }
                    else if (var == "eps")
                    {
                        file >> c.epsilon;
                    }
                    else if (var == "tau")
                    {
                        file >> c.tau;
                    }
                    else if (var == "gamma")
                    {
                        file >> c.gamma;
                    }
                    else if (var == "dt_value")
                    {
                        file >> c.output_freq;
                    }
                    else if (var == "UI")
                    {
                        file >> c.UI;
                    }
                    else if (var == "VI")
                    {
                        file >> c.VI;
                    }
                    else if (var == "GX")
                    {
                        file >> c.GX;
                    }
                    else if (var == "GY")
                    {
                        file >> c.GY;
                    }
                    else if (var == "PI")
                    {
                        file >> c.PI;
                    }
                    else if (var == "itermax")
                    {
                        file >> c.itermax;
                    }
                    else if (var == "geo_file")
                    {
                        file >> c.geom_file;
                    }
                    else if (var == "energy_eq")
                    {
                        std::string str;
                        file >> str;
                        c.energy_eq = str == "on";
                    }
                    else if (var == "top_bc_type")
                    {
                        file >> c.top_bc.type;
                    }
                    else if (var == "top_bc_value")
                    {
                        file >> c.top_bc.value;
                    }
                    else if (var == "bottom_bc_type")
                    {
                        file >> c.bottom_bc.type;
                    }
                    else if (var == "bottom_bc_value")
                    {
                        file >> c.bottom_bc.value;
                    }
                    else if (var == "left_bc_type")
                    {
                        file >> c.left_bc.type;
                    }
                    else if (var == "left_bc_value")
                    {
                        file >> c.left_bc.value;
                    }
                    else if (var == "right_bc_type")
                    {
                        file >> c.right_bc.type;
                    }
                    else if (var == "right_bc_value")
                    {
                        file >> c.right_bc.value;
                    }
                    else if (var == "reconstruction_frequency")
                    {
                        file >> c.reconstruction_freq;
                    }
                    else if (var == "vorticity_lower_limit")
                    {
                        file >> c.vorticity_lower;
                    }
                    else if (var == "vorticity_upper_limit")
                    {
                        file >> c.vorticity_upper;
                    }
                    else if (var == "max_tree_depth")
                    {
                        file >> c.max_tree_depth;
                    }
                    else if (var == "initial_min_refinement")
                    {
                        file >> c.init_refinement;
                    }
                    else
                    {
                        assert(false);
                    }
                }
            }
        }
        file.close();

        return c;
    }

    // Read input parameters
    double t_end{};
    double nu{};      /* viscosity   */
    double alpha{};   /* thermal diffusity   */
    double beta{};    /* thermal expansion coefficient  */
    double TI{};      /* temperature  */
    double UI{};      /* velocity x-direction */
    double VI{};      /* velocity y-direction */
    double PI{};      /* pressure */
    double GX{};      /* gravitation x-direction */
    double GY{};      /* gravitation y-direction */
    double xlength{}; /* length of the domain x-dir.*/
    double ylength{}; /* length of the domain y-dir.*/
    double dt{};      /* time step */
    double gamma{};   /* uppwind differencing factor*/
    double omega{};   /* relaxation factor */
    double tau{};     /* safety factor for time step*/
    int itermax{};    /* max. number of iterations for pressure per time step */
    double epsilon{}; /* accuracy bound for pressure*/
    std::string geom_file = "NONE"; /*name of geometry file*/
    bool energy_eq = false;         /*include energy equation (on/off)*/
    double output_freq{};           /*output frequency*/
    PhysicalBC top_bc{};            /*top boundary condition*/
    PhysicalBC bottom_bc{};         /*bottom boundary condition*/
    PhysicalBC right_bc{};          /*right boundary condition*/
    PhysicalBC left_bc{};           /*left boundary condition*/
    double reconstruction_freq{};   /*reconstruction frequency of the tree */
    double vorticity_lower{}; /*lower vorticity limit for coarsening condition*/
    double vorticity_upper{}; /*upper vorticity limit for refinement condition*/
    unsigned int max_tree_depth{};  /*maximum deth (0 -> default)*/
    unsigned int init_refinement{}; /*initial refinemnt of the tree*/
};

inline Matrix<cell_type> read_pgm_file(const std::string& path)
{

    std::ifstream file(path);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open PGM file: " + path);
    }

    int square_length = 0;
    std::string line;

    while (std::getline(file, line))
    {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        iss >> square_length;
        if (square_length > 0) break;
    }

    Matrix<cell_type> pgm_data(
        square_length, square_length, cell_type::DEFAULT
    );

    int y = 0;
    int x = 0;
    while (file >> line)
    {
        for (size_t i = 0; i < line.size(); ++i)
        {
            int val = line[i] - '0';
            pgm_data(x, y) =
                (val == 0) ? cell_type::FLUID : cell_type::OBSTACLE;
            if (++x == square_length)
            {
                x = 0;
                if (++y == square_length) goto end;
            }
        }
    }
end:
    return pgm_data;
}
