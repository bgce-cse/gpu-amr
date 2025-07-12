module TerraDG
using WriteVTK
using Printf
using Logging
using LinearAlgebra
import YAML

include("configuration.jl")
include("basis.jl")
include("equations.jl")
include("structure/mesh_structure.jl")
include("grid.jl")
include("kernels/surface.jl")
include("kernels/volume.jl")
include("kernels/filtering.jl")
include("kernels/time.jl")
include("kernels/limiting.jl")
include("plotters.jl")
include("error_writer.jl")
include("global_matrices.jl")
include("structure/cellarray.jl")      # Include this FIRST
include("structure/amr_quad_tree.jl")  # Then include this



"""
    evaluate_rhs(eq, scenario, filter, globals, du, dofs, grid)

Evalutes the right-hand-side of the equation `eq` for
scenario `scenario`, with filter `filter`, 
collection of global matrices `globals`, update
`du`, degrees of freedom `dofs` and grid `grid`.

Updates `du` in place.
"""
function evaluate_rhs(eq, scenario, filter, globals, du, dofs, grid)
    buffers_face = BuffersFaceIntegral(grid.basis, get_ndofs(eq))
    buffers_volume = BuffersVolume(grid.basis, get_ndofs(eq))

    nvar = get_ndofs(eq)
    reference_massmatrix = massmatrix(grid.basis, grid.basis.dimensions)
    du .= 0.0
    maxeigenval = -Inf


    ∇ = globals.reference_derivative_matrix
    for i in eachindex(grid.cells)
        @views cell = grid.cells[i]
        @views data = dofs[:,:, cell.dataidx]
        @views flux = grid.flux[:,:,cell.dataidx]        
        evaluate_flux(eq, data, flux)

        # Volume matrix is zero for FV/order=1
        if length(grid.basis.quadpoints) > 1
            @views evaluate_volume(globals, buffers_volume, flux, grid.basis, inverse_jacobian(cell), volume(cell), du[:,:,cell.dataidx])
            #print("volume: ",norm(du))
        end
    end
    for i in eachindex(grid.cells)
        @views cell = grid.cells[i]
        @views data = dofs[:,:, cell.dataidx]
        @views flux = grid.flux[:,:,cell.dataidx]
        elem_massmatrix = volume(cell) * reference_massmatrix
        inv_massmatrix = inv(elem_massmatrix)

        # Here we also need to compute the maximum eigenvalue of each cell
        # and store it for each cell (needed for timestep restriction later!)
        faces = [W, N, E, S]
        for (i, neigh) in enumerate(cell.neighbors)
            if !isempty(cell.neighbors[Face(i)])
                @views dofsneigh = dofs[:,:,cell.neighbors[Face(i)][1].dataidx]
                @views fluxneigh = grid.flux[:,:,cell.neighbors[Face(i)][1].dataidx]
            
                # Project dofs and flux of own cell to face
                project_to_faces(globals, data, flux, buffers_face.dofsface, buffers_face.fluxface, faces[i])
            end
            facetypeneigh = cell.facetypes[i] #TODO until here it works for both
            
            if facetypeneigh == regular 
            
            
                # Project neighbors to faces
                # Neighbor needs to project to opposite face
                faceneigh = globals.oppositefaces[faces[i]]
                project_to_faces(globals, dofsneigh, fluxneigh, buffers_face.dofsfaceneigh, buffers_face.fluxfaceneigh, faceneigh)
            else
                @assert(facetypeneigh == boundary)
                normalidx = globals.normalidxs[faces[i]]

                # For boundary cells, we operate directly on the dofsface
                evaluate_boundary(eq, scenario, faces[i], normalidx, buffers_face.dofsface, buffers_face.dofsfaceneigh)
                # Evaluate flux on face directly
                # Note: When extrapolating, this is not exact!
                # The error is given by the commutation error of face projection and flux!
                evaluate_flux(eq, buffers_face.dofsfaceneigh, buffers_face.fluxfaceneigh)
            end

            @views cureigenval = evaluate_face_integral(eq, globals, buffers_face, cell, faces[i], du[:,:,cell.dataidx])
            
            maxeigenval = max(maxeigenval, cureigenval)
            print(cureigenval," ", maxeigenval,"\n")
            #print("faces $(faces[i]): ",norm(du))
        end
        @views du[:,:,cell.dataidx] = inv_massmatrix * @views du[:,:,cell.dataidx] 
            #print("mass: ",norm(du))
    end
    grid.maxeigenval = maxeigenval
end

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
        @views interpolate_initial_dofs(eq, scenario, mesh_struct.dofs[:,:,i],mesh_struct.cells[i],mesh_struct.basis)
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
            print(dt, next_plotted-mesh_struct.time, config.end_time - mesh_struct.time)
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
            for cell in mesh_struct.cells
                @views celldata = mesh_struct.dofs[:,:,cell.dataidx] 
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