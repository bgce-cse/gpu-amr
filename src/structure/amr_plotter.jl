using LinearAlgebra
using WriteVTK
using WriteVTK: paraview_collection, collection_add_timestep # Explicitly import these
using Printf # For @sprintf

# Define the base output directory for all VTK files
const output_base_dir = "output"

# NOTE: The `pointidx` and `get_plot_points` functions are for high-order elements.
# The AMRQuadTree, as described, uses simple rectangular cells (VTK_QUAD).
# If you intend to use high-order elements *within* each AMR cell,
# you would need to integrate `get_plot_points` into the `plot` function
# and adjust how `vtkpoints_matrix` and `vtkcells_list` are constructed.
# For now, this code assumes each AMR leaf node is a simple VTK_QUAD.

mutable struct AMRVTKPlotter
    filename_base_pvd::String # Full path for the .pvd file (e.g., "output/my_amr_plot.pvd")
    collection::WriteVTK.CollectionFile
    plot_counter::Int64
end

"""
Inizializza AMRVTKPlotter con il nome base del file pvd.
`filename_stem` is just the base name without directory or extension, e.g., "my_amr_plot".
The actual output will be in the `output_base_dir`.
"""
function AMRVTKPlotter(filename_stem::String)
    # Ensure the base output directory exists
    if !isdir(output_base_dir)
        mkpath(output_base_dir)
    end

    # The full path for the .pvd file
    filename_base_pvd_full = joinpath(output_base_dir, filename_stem)
    
    collection = paraview_collection(filename_base_pvd_full) 
    plot_counter = 0
    return AMRVTKPlotter(filename_base_pvd_full, collection, plot_counter)
end

"""
Funzione per creare il file vtu e aggiornare la collezione pvd.
`tree` è l’oggetto AMRQuadTree con i nodi foglia da plottare.
"""
function plot(plotter::AMRVTKPlotter, tree)
    leaf_nodes = get_leaf_nodes(tree)

    # --- Dynamically generate points and cells for the current leaf nodes ---
    unique_points_set = Set{Tuple{Float64, Float64, Float64}}()
    
    for node in leaf_nodes
        push!(unique_points_set, (node.x, node.y, 0.0))
        push!(unique_points_set, (node.x + node.size, node.y, 0.0))
        push!(unique_points_set, (node.x, node.y + node.size, 0.0))
        push!(unique_points_set, (node.x + node.size, node.y + node.size, 0.0))
    end

    unique_points_list = collect(unique_points_set)
    sort!(unique_points_list) # Sort for deterministic VTK output
    
    points_map = Dict(p => i for (i, p) in enumerate(unique_points_list))

    n_unique_points = length(unique_points_list)
    vtkpoints_matrix = zeros(3, n_unique_points)
    for (i, p) in enumerate(unique_points_list)
        vtkpoints_matrix[:, i] = [p[1], p[2], p[3]]
    end

    vtkcells_list = MeshCell[]
    cell_levels = Int[] # Data to store per cell
    cell_ids = Int[]    # Data to store per cell

    for node in leaf_nodes
        p1 = (node.x, node.y, 0.0)
        p2 = (node.x + node.size, node.y, 0.0)
        p3 = (node.x + node.size, node.y + node.size, 0.0)
        p4 = (node.x, node.y + node.size, 0.0)
        
        # VTK_QUAD expects points in counter-clockwise order: (p1, p2, p3, p4) for bottom-left, bottom-right, top-right, top-left
        push!(vtkcells_list, MeshCell(VTKCellTypes.VTK_QUAD, [points_map[p1], points_map[p2], points_map[p3], points_map[p4]]))
        push!(cell_levels, node.level)
        push!(cell_ids, node.id)
    end
    # --- End dynamic generation ---

    # Increment counter BEFORE using it for the current plot's filename and time
    plotter.plot_counter += 1 

    # Construct the local filename for the .vtu file (e.g., "my_amr_plot_0001.vtu")
    # basename(plotter.filename_base_pvd) will extract "my_amr_plot" from "output/my_amr_plot"
    vtu_filename_local = @sprintf("%s_%04d.vtu", basename(plotter.filename_base_pvd), plotter.plot_counter)

    # Construct the full path where the .vtu file will be saved
    vtu_filepath_full = joinpath(output_base_dir, vtu_filename_local)

    # Create VTK file
    vtkfile = vtk_grid(vtu_filepath_full, vtkpoints_matrix, vtkcells_list)

    # Add cell data
    vtk_cell_data(vtkfile, cell_levels, "Refinement_Level")
    vtk_cell_data(vtkfile, cell_ids, "Node_ID")

    # Save the VTK file
    vtk_save(vtkfile)

    # Add to the PVD collection using collection_add_timestep
    # The path for the PVD entry must be relative to the PVD file itself.
    # Since both .pvd and .vtu files are in `output_base_dir`, just the .vtu filename is needed.
    # We use plotter.plot_counter * 0.1 as a dummy time value, or you could use tree.time if available.
    collection_add_timestep(plotter.collection, vtkfile, plotter.plot_counter * 0.1) 

    println("Saved VTU: $vtu_filepath_full with $(length(leaf_nodes)) cells.")
    return vtu_filepath_full
end

"""
Salva il file PVD finale
"""
function save(plotter::AMRVTKPlotter)
    vtk_save(plotter.collection)
    println("Saved collection PVD file to $(plotter.filename_base_pvd).pvd.")
end