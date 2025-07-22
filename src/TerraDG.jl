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
include("structure/grid.jl")
include("kernels/surface.jl")
include("kernels/volume.jl")
include("kernels/filtering.jl")
include("kernels/time.jl")
include("kernels/limiting.jl")
include("plotters.jl")
include("error_writer.jl")
include("global_matrices.jl")
include("structure/cellarray.jl")     
include("structure/amr_quad_tree.jl") 

"""
evaluate_rhs(eq, scenario, filter, globals, du, dofs, grid)

Evalutes the right-hand-side of the equation `eq` for
scenario `scenario`, with filter `filter`, 
collection of global matrices `globals`, update
`du`, degrees of freedom `dofs` and grid `grid`.

Updates `du` in place.
"""
function evaluate_rhs(eq, scenario, basis, filter, globals, du, dofs, grid)
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
        end
    end
    gradients = Array{Float64}(undef, length(grid.cells))
    for i in eachindex(grid.cells)
        @views cell = grid.cells[i]
        @views data = dofs[:,:, cell.dataidx]
        @views flux = grid.flux[:,:,cell.dataidx]
        elem_massmatrix = volume(cell) * reference_massmatrix
        inv_massmatrix = inv(elem_massmatrix)

        # Here we also need to compute the maximum eigenvalue of each cell
        # and store it for each cell (needed for timestep restriction later!)
        for (i, neigh) in enumerate(cell.neighbors)
            
            for (j, sub_neigh) in enumerate(cell.neighbors[Face(i)])
                @views dofsneigh = dofs[:,:,sub_neigh.dataidx]
                @views fluxneigh = grid.flux[:,:,sub_neigh.dataidx]
            
                # Project dofs and flux of own cell to face
                project_to_faces(globals, data, flux, buffers_face.dofsface, buffers_face.fluxface, Face(i))

                facetypeneigh = cell.facetypes[i]
                
                if facetypeneigh == regular
                    faceneigh = globals.oppositefaces[Face(i)]
    
                    if volume(cell) > volume(sub_neigh)
                        # Neighbor is smaller than current cell
                        project_smaller_neighbor!(basis, cell, sub_neigh, dofsneigh, fluxneigh, 
                                                buffers_face, faceneigh, i)
                    elseif volume(cell) < volume(sub_neigh)
                        # Neighbor is larger than current cell
                        project_larger_neighbor!(basis, cell, sub_neigh, dofsneigh, fluxneigh, 
                                            buffers_face, faceneigh, i)
                    else 
                        # Same size neighbors
                        project_to_faces(globals, dofsneigh, fluxneigh, 
                                        buffers_face.dofsfaceneigh, buffers_face.fluxfaceneigh, faceneigh)
                    end

                else
                    @assert(facetypeneigh == boundary)

                    normalidx = globals.normalidxs[Face(i)]

                    # For boundary cells, we operate directly on the dofsface
                    evaluate_boundary(eq, scenario, Face(i), normalidx, buffers_face.dofsface, buffers_face.dofsfaceneigh)

                    evaluate_flux(eq, buffers_face.dofsfaceneigh, buffers_face.fluxfaceneigh)
                end
            end
                @views cureigenval = evaluate_face_integral(eq, globals, buffers_face, cell, Face(i), du[:,:,cell.dataidx]) 
                maxeigenval = max(maxeigenval, cureigenval)
        
        end
        @views du[:,:,cell.dataidx] = inv_massmatrix * @views du[:,:,cell.dataidx] 

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
    
    @info "Initialising global matrices"
    globals = GlobalMatrices(mesh_struct.basis, filter, mesh_struct.basis.dimensions)
    @info "Initialised global matrices"
    
    filename = "output/plot"
    
    for i in eachindex(mesh_struct.cells)
        @views interpolate_initial_dofs(eq, scenario, mesh_struct.dofs[:,:,i],mesh_struct.cells[i],mesh_struct.basis)
    end
    
    
    mesh_struct.time = 0
    timestep = 0

    global plotter = VTKPlotter(eq, scenario, mesh_struct, filename)
    

    next_plotted = config.plot_step
    mesh_changed = false
    

    @info "Plotting initial values (unrefined)"
    plot(plotter)
    

    if amr

        @info "Refining mesh"
        refined = amr_update!(mesh_struct, eq, scenario, globals,config)
        if refined
            mesh_changed = true
            @info "Mesh was refined"
        end
        

        if mesh_changed
            @info "Updating plotter due to mesh refinement"
            update_plotter!(plotter, mesh_struct)
            mesh_changed = false
        end
        @info "Plotting initial values (refined)"
        plot(plotter)
    end
    

    @info "Starting timestepping"
    while mesh_struct.time < config.end_time

        if timestep % 20 == 19 && amr # Refine every 20 timesteps
            refined = amr_update!(mesh_struct, eq, scenario,globals, config)
            if refined 
                mesh_changed = true
            end
        end

        integrator = make_timeintegrator(config, mesh_struct)

        if timestep > 0
            time_start = time()

            
            dt = 1/(config.order^2+1) * (config.physicalsize[1]/(2^mesh_struct.current_refinement_level)) * config.courant * 1/mesh_struct.maxeigenval

            # Only step up to either end or next plotting
            dt = min(dt, next_plotted-mesh_struct.time, config.end_time - mesh_struct.time)
            @assert dt > 0
            @info "Running timestep" timestep dt mesh_struct.time
            @info "Current refinement level" mesh_struct.current_refinement_level

            step(integrator, mesh_struct, dt) do du, dofs, time
                evaluate_rhs(eq, scenario, mesh_struct.basis, filter, globals, du, dofs, mesh_struct)
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

        # Check if it's time to plot
        if mesh_struct.time >= next_plotted - 1e-10
            @info "Writing output" mesh_struct.time next_plotted

            try
                # If mesh changed, update the plotter
                if mesh_changed
                    @info "Updating plotter due to mesh refinement"
                    update_plotter!(plotter, mesh_struct)
                    mesh_changed = false
                end

                # Plot current state
                plot(plotter)
                @info "Successfully wrote VTK output for timestep" plotter.plot_counter-1

            catch e
                @error "Failed to write VTK output" exception=e
                println("Error details: ", e)
                # Print the stack trace
                for (exc, bt) in Base.catch_stack()
                    showerror(stdout, exc, bt)
                    println()
                end
            end

            next_plotted = mesh_struct.time + config.plot_step

            # TODO delete if want to solve sibson for time
            if config.equation_name == "sibson" 
                break
            end
        end

        timestep += 1
    end

    # Final save
    try
        save(plotter)
        @info "Final save completed"
    catch e
        @error "Failed to save final output" exception=e
    end

    if is_analytical_solution(eq, scenario)
        evaluate_error(eq, scenario, mesh_struct, mesh_struct.time)
    end

end
end