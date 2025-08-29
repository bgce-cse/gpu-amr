#include "Case.hpp"
#include "boundary.hpp"
#include "config.hpp"
#include "domain.hpp"
#include "tree_types.hpp"
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
namespace fs = std::filesystem;

int main(int argn, char** args)
{
    if (argn < 1)
    {
        std::cout << "Error: No input file is provided to fluidchen."
                  << std::endl;
        std::cout << "Example usage: /path/to/fluidchen /path/to/input_data.dat"
                  << std::endl;
    }

    std::string file_name{args[1]};

    const auto config = sim_config::parse_config(file_name, argn, args);

    Matrix<cell_type> pgm_matrix;
    Matrix<cell_type> const* pgm_data = nullptr;
    if (config.geom_file != "NONE")
    {
        fs::path dat_path(file_name);
        fs::path pgm_path = dat_path.parent_path() / config.geom_file;
        pgm_matrix = read_pgm_file(pgm_path);
        pgm_data = &pgm_matrix;
    }

    BCSet top_bc_set = map_physical_bc(config.top_bc);
    BCSet left_bc_set = map_physical_bc(config.left_bc);
    BCSet right_bc_set = map_physical_bc(config.right_bc);
    BCSet bottom_bc_set = map_physical_bc(config.bottom_bc);

    Fields field(
        config.nu,
        config.GX,
        config.GY,
        config.dt,
        config.tau,
        config.alpha,
        config.beta,
        config.energy_eq
    );

    const SOR pressure_solver(config.omega);
    Discretization::set_gamma(config.gamma);

    qoi_boundary<u_t>::domain_bc u_bc{
        .top = {top_bc_set.u_bc_type, top_bc_set.u_bc_value},
        .left = {left_bc_set.u_bc_type, left_bc_set.u_bc_value},
        .right = {right_bc_set.u_bc_type, right_bc_set.u_bc_value},
        .bottom = {bottom_bc_set.u_bc_type, bottom_bc_set.u_bc_value},
        .internal = {BoundaryType::Dirichlet, 0.0}
    };
    qoi_boundary<v_t>::domain_bc v_bc{
        .top = {top_bc_set.v_bc_type, top_bc_set.v_bc_value},
        .left = {left_bc_set.v_bc_type, left_bc_set.v_bc_value},
        .right = {right_bc_set.v_bc_type, right_bc_set.v_bc_value},
        .bottom = {bottom_bc_set.v_bc_type, bottom_bc_set.v_bc_value},
        .internal = {BoundaryType::Dirichlet, 0.0}
    };
    qoi_boundary<p_t>::domain_bc p_bc{
        .top = {top_bc_set.p_bc_type, top_bc_set.p_bc_value},
        .left = {left_bc_set.p_bc_type, left_bc_set.p_bc_value},
        .right = {right_bc_set.p_bc_type, right_bc_set.p_bc_value},
        .bottom = {bottom_bc_set.p_bc_type, bottom_bc_set.p_bc_value},
        .internal = {BoundaryType::Neumann, 0.0}
    };
    qoi_boundary<t_t>::domain_bc t_bc{
        .top = {top_bc_set.t_bc_type, top_bc_set.t_bc_value},
        .left = {left_bc_set.t_bc_type, left_bc_set.t_bc_value},
        .right = {right_bc_set.t_bc_type, right_bc_set.t_bc_value},
        .bottom = {bottom_bc_set.t_bc_type, bottom_bc_set.t_bc_value},
        .internal = {BoundaryType::Neumann, 0.0}
    };

    domain_bc bc{.u_bc = u_bc, .v_bc = v_bc, .p_bc = p_bc, .t_bc = t_bc};

    index_t::set_max_depth(config.max_tree_depth);
    sim_domain domain(
        100000u,
        cell(config.UI, config.VI, 0, 0, config.PI, config.TI, 0),
        config.init_refinement,
        config.xlength,
        config.ylength,
        bc,
        pgm_data
    );

    Case problem(
        file_name,
        config.itermax,
        config.epsilon,
        config.output_freq,
        config.reconstruction_freq,
        config.vorticity_lower,
        config.vorticity_upper
    );
    problem.simulate(domain, field, pressure_solver, config.t_end);

    return EXIT_SUCCESS;
}
