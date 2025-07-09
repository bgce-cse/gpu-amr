module TerraDG
using WriteVTK
using Printf
using Logging
using LinearAlgebra
import YAML

include("configuration.jl")
include("basis.jl")
include("equations.jl")
include("grid.jl")
include("kernels/surface.jl")
include("kernels/volume.jl")
include("kernels/filtering.jl")
include("kernels/time.jl")
include("kernels/limiting.jl")
include("plotters.jl")
include("error_writer.jl")
include("structure/mesh_structure.jl")
include("global_matrices.jl")
include("structure/amr_quad_tree.jl")

function main(configfile::String)
    config = Configuration(configfile)
    filter = make_filter(config)
    eq = make_equation(config)
    scenario = make_scenario(config)
    amr = config.amr

    mesh_struct =  create_struct(amr, config, eq, scenario)
    integrator = make_timeintegrator(config, mesh_struct)

    @info "Initialising global matrices"
    globals = GlobalMatrices(mesh_struct.basis, filter, mesh_struct.basis.dimensions)
    @info "Initialised global matrices"

    filename = "output/plot"

    for i in eachindex(mesh_struct.cells)
        @views interpolate_initial_dofs(eq, scenario, mesh_struct.dofs[:,:,i],mesh_struct.cell[i],mesh_struct.basis)
    end
    
    plotter = VTKPlotter(eq, scenario, mesh_struct, filename) #TODO change plotter

    mesh_struct.time = 0
    timestep = 0
    next_plotted = config.plot_start

    #limiter? 

    while mesh_struct.time < config.end_time
        if timestep > 0
            time_start = time()
            dt = 1/(config.order^2+1) * config.cellsize[1] * config.courant * 1/mesh_struct.maxeigenval
            
            #limiter?
            # Only step up to either end or next plotting
            dt = min(dt, next_plotted-mesh_struct.time, config.end_time - mesh_struct.time)
            @assert dt > 0
            @info "Running timestep" timestep dt mesh_struct.time
            step(integrator, mesh_struct, dt) do du, dofs, time
                evaluate_rhs(eq, scenario, filter, globals, du, dofs, mesh_struct)#TODO figure out what to do here
            end
            mesh_struct.time += dt
            time_end = time()
            time_elapsed = time_end - time_start
            @info "Timestep took" time_elapsed
        else
            # Compute initial eigenvalue (needed for dt)
            mesh_struct.maxeigenval = -1
            for i in each_index(mesh_struct.cells)
                @views celldata = mesh_struct.dofs[:,:,i] 
                for normalidx=1:2
                    cureigenval = max_eigenval(eq, celldata, normalidx)
                    mesh_struct.maxeigenval = max(mesh_struct.maxeigenval, cureigenval)

                end
            end
        end
        if abs(mesh_struct.time - next_plotted) < 1e-10
            #limiter?
            @info "Writing output" mesh_struct.time
            plot(plotter)
            next_plotted = mesh_struct.time + config.plot_step

            
            #TODO delete if want to solve sibson for time
            if config.equation_name == "sibson" 
                break
            end
        end
        timestep += 1
    end
    save(plotter)
    if is_analytical_solution(eq, scenario)
        evaluate_error(eq, scenario, mesh_struct, mesh_struct.time)
    end
end
end