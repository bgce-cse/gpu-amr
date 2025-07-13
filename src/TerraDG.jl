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


function refine_random_node!(tree::AMRQuadTree)
    leaf_nodes = get_leaf_nodes(tree)
    
    if isempty(leaf_nodes)
        println("No leaf nodes to refine")
        return
    end
    
    # Try to refine a random node
    max_attempts = 10
    for attempt in 1:max_attempts
        node = rand(leaf_nodes)
        if refine_node!(tree, node)
            println("Successfully refined random node $(node.id) at level $(node.level)")
            println("New number of leaf nodes: ", length(get_leaf_nodes(tree)))
            return
        end
    end
    
    println("Could not refine any random node after $max_attempts attempts")
end

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
    for i in eachindex(grid.cells)
        @views cell = grid.cells[i]
        @views data = dofs[:,:, cell.dataidx]
        @views flux = grid.flux[:,:,cell.dataidx]
        elem_massmatrix = volume(cell) * reference_massmatrix
        inv_massmatrix = inv(elem_massmatrix)

        # Here we also need to compute the maximum eigenvalue of each cell
        # and store it for each cell (needed for timestep restriction later!)
        faces = [W, N, E, S]
        #print("cell $(cell.dataidx) size :$(cell.size)\n")
        for (i, neigh) in enumerate(cell.neighbors)
            
            for (j, sub_neigh) in enumerate(cell.neighbors[Face(i)])
                # print("neigh $(i) : $(sub_neigh.id) : $(sub_neigh.center)\n")
                @views dofsneigh = dofs[:,:,sub_neigh.dataidx]
                @views fluxneigh = grid.flux[:,:,sub_neigh.dataidx]
            
                # Project dofs and flux of own cell to face
                project_to_faces(globals, data, flux, buffers_face.dofsface, buffers_face.fluxface, faces[i])

                facetypeneigh = cell.facetypes[i]
                
                if facetypeneigh == regular
                    # Project neighbors to faces
                    # Neighbor needs to project to opposite face
                    faceneigh = globals.oppositefaces[faces[i]]
                    if volume(cell) > volume(sub_neigh)
                        face_points = get_face_quadpoints(basis, faceneigh)
                        if Face(i) == N || Face(i) == S
                            mask = (cell.center[1] < sub_neigh.center[1]) ? (face_points[1] .>= 0.5) : (face_points[1] .< 0.5)
                            offset = (cell.center[1] < sub_neigh.center[1]) ? -1.0 : 0.0
                            masked_indices = findall(mask)
                            points = (face_points[1] .* 2 .+ offset, face_points[2])

                            for (masked_idx, real_idx) in enumerate(masked_indices)
                                # print("NS $(sub_neigh.dataidx): $(masked_idx), $(real_idx) ,$(offset)\n")
                                buffers_face.dofsfaceneigh[real_idx, :] = evaluate_basis(basis, dofsneigh, [points[1][real_idx] points[2]])
                            end

                            for (masked_idx, real_idx) in enumerate(masked_indices)
                                buffers_face.fluxfaceneigh[real_idx, :] = evaluate_basis(basis, fluxneigh[1:size(fluxneigh,1)÷2, :], [points[1][real_idx], points[2]])
                                buffers_face.fluxfaceneigh[real_idx + basis.order, :] = evaluate_basis(basis, fluxneigh[size(fluxneigh,1)÷2+1:end, :], [points[1][real_idx], points[2]])
                            end

                        else
                            mask = (cell.center[2] < sub_neigh.center[2]) ? (face_points[2] .>= 0.5) : (face_points[2] .< 0.5)
                            offset = (cell.center[2] < sub_neigh.center[2]) ? -1.0 : 0.0
                            masked_indices = findall(mask)
                            points = (face_points[1], face_points[2] .* 2 .+ offset)

                            for (masked_idx, real_idx) in enumerate(masked_indices)
                                # print("WE  $(sub_neigh.dataidx): $(masked_idx), $(real_idx),$(offset) \n")
                                buffers_face.dofsfaceneigh[real_idx, :] = evaluate_basis(basis, dofsneigh, [points[1], points[2][real_idx]])
                            end

                            for (masked_idx, real_idx) in enumerate(masked_indices)
                                buffers_face.fluxfaceneigh[real_idx, :] = evaluate_basis(basis, fluxneigh[1:size(fluxneigh,1)÷2, :], [points[1], points[2][real_idx] ])
                                buffers_face.fluxfaceneigh[real_idx + basis.order, :] = evaluate_basis(basis, fluxneigh[size(fluxneigh,1)÷2+1:end, :], [points[1], points[2][real_idx]])
                            end
                        end
                        
                    elseif volume(cell) < volume(sub_neigh)
                    
                        face_points = get_face_quadpoints(basis, faceneigh)

                        if Face(i) == N || Face(i) == S
                            # North/South faces
                            offset = (cell.center[1] < sub_neigh.center[1]) ? 0.0 : 0.5
                            points = (face_points[1] ./ 2 .+ offset, face_points[2])
                            #print("NS $(sub_neigh.dataidx): $(points[1]), $(offset)\n")
                            # Evaluate basis functions and copy into existing matrix
                            for idx in 1:basis.order
                                buffers_face.dofsfaceneigh[idx, :] = evaluate_basis(basis, dofsneigh, [points[1][idx], points[2]])
                            end
                            
                            # Evaluate fluxes and copy into existing matrix
                            for idx in 1:basis.order
                                buffers_face.fluxfaceneigh[idx, :] = evaluate_basis(basis, fluxneigh[1:size(fluxneigh,1)÷2, :], [points[1][idx], points[2]])
                                buffers_face.fluxfaceneigh[idx + basis.order, :] = evaluate_basis(basis, fluxneigh[size(fluxneigh,1)÷2+1:end, :], [points[1][idx], points[2]])
                            end
                        else
                            # East/West faces
                            offset = (cell.center[2] < sub_neigh.center[2]) ? 0.0 : 0.5
                            points = (face_points[1], face_points[2] ./ 2 .+ offset)
                            #print("WE $(sub_neigh.dataidx): $(points[2]), $(offset)\n")
                            # Evaluate basis functions and copy into existing matrix
                            for idx in 1:basis.order
                                buffers_face.dofsfaceneigh[idx, :] = evaluate_basis(basis, dofsneigh, [points[1], points[2][idx]])
                            end
                            
                            # Evaluate fluxes and copy into existing matrix
                            for idx in 1:basis.order
                                buffers_face.fluxfaceneigh[idx, :] = evaluate_basis(basis, fluxneigh[1:size(fluxneigh,1)÷2, :], [points[1], points[2][idx]])
                                buffers_face.fluxfaceneigh[idx + basis.order, :] = evaluate_basis(basis, fluxneigh[size(fluxneigh,1)÷2+1:end, :], [points[1], points[2][idx]])
                            end
                        end
                    else 
                        project_to_faces(globals, dofsneigh, fluxneigh, buffers_face.dofsfaceneigh, buffers_face.fluxfaceneigh, faceneigh)
                    end

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
            end
                @views cureigenval = evaluate_face_integral(eq, globals, buffers_face, cell, faces[i], du[:,:,cell.dataidx])
                
                maxeigenval = max(maxeigenval, cureigenval)
                #print(cureigenval," ", maxeigenval,"\n")
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

    # if config.amr
    #     # refine_node!(mesh_struct, )
    #     refine_random_node!(mesh_struct)
    # end

    
    @info "Initialising global matrices"
    globals = GlobalMatrices(mesh_struct.basis, filter, mesh_struct.basis.dimensions)
    @info "Initialised global matrices"
    
    filename = "output/plot"
    
    for i in eachindex(mesh_struct.cells)
        @views interpolate_initial_dofs(eq, scenario, mesh_struct.dofs[:,:,i],mesh_struct.cells[i],mesh_struct.basis)
    end
    
    
    mesh_struct.time = 0
    timestep = 0
    # Create plotter ONCE before the loop
    global plotter = VTKPlotter(eq, scenario, mesh_struct, filename)
    
    # Initialize plotting variables
    next_plotted = config.plot_step
    mesh_changed = false
    
    # Stage 1: Plot initial values unrefined
    @info "Plotting initial values (unrefined)"
    plot(plotter)
    
    # Stage 2: Refine mesh
    @info "Refining mesh"
    refined = amr_refine!(mesh_struct, eq, scenario)
    if refined
        mesh_changed = true
        @info "Mesh was refined"
    end
    
    # Stage 3: Plot initial values refined
    if mesh_changed
        @info "Updating plotter due to mesh refinement"
        update_plotter!(plotter, mesh_struct)
        mesh_changed = false
    end
    @info "Plotting initial values (refined)"
    plot(plotter)
    
    # Stage 4: Start timestepping
    @info "Starting timestepping"
    while mesh_struct.time < config.end_time

        if timestep == 100 # Refine every 50 timesteps
            refined = amr_refine!(mesh_struct, eq, scenario)
            if refined  # Assuming amr_refine! returns true if mesh was actually refined
                mesh_changed = true
            end
        end

        integrator = make_timeintegrator(config, mesh_struct)

        if timestep > 0
            time_start = time()

            print(mesh_struct.current_refinement_level)
            dt = 1/(config.order^2+1) * (config.physicalsize[1]/(2^mesh_struct.current_refinement_level)) * config.courant * 1/mesh_struct.maxeigenval

            # Only step up to either end or next plotting
            dt = min(dt, next_plotted-mesh_struct.time, config.end_time - mesh_struct.time)
            @assert dt > 0
            @info "Running timestep" timestep dt mesh_struct.time

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