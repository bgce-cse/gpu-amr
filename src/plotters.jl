function pointidx(i::Integer, j::Integer, order)
    # I'm sorry
    # Quad: https://gitlab.kitware.com/nick.laurenson/vtk/-/blob/d8aa5b89c622cf04b3322112fda420afc9d2a16d/Common/DataModel/vtkLagrangeQuadrilateral.cxx#L558
    # Hex: https://gitlab.kitware.com/nick.laurenson/vtk/blob/d8aa5b89c622cf04b3322112fda420afc9d2a16d/Common/DataModel/vtkLagrangeHexahedron.cxx#L734
    # Implemented for quad here
    is_i_boundary = i == 0 || i == order[1]
    is_j_boundary = j == 0 || j == order[2]
    
    n_boundary = is_i_boundary + is_j_boundary
    
    i_pos = i > 0
    j_pos = j > 0
    
    if n_boundary == 2
        # Vertex DOF
        return (i_pos ? (j_pos ? 2 : 1) : (j_pos ? 3 : 0))
    end
    
    offset = 4
    if n_boundary == 1 
        # Edge DOF
        if !is_i_boundary
            # On i axis
            return (i - 1) +
            (j_pos ? order[1] - 1 + order[2] - 1 : 0) +
            offset
        end
        if !is_j_boundary
            # On j axis
            return (j - 1) +
            (i_pos ? order[1] - 1 : 2 * (order[1] - 1) + order[2] - 1) +
            offset
        end
    end

      offset += 2 * (order[1] - 1 + order[2] - 1)
      # n_boundary == 0 -> Face DOF
      return offset +
        (i - 1) + (order[1] - 1) * (
          (j - 1))
end

function get_plot_points(order)
    dim = 2
    n_points_x = (order + 1)
    n_points = (order + 1) * (order + 1)
    n_points_x, n_points
    points = zeros(dim, n_points)
    #xs = range(0.0, 1.0, length=n_points_x)
    xs = (gausslobatto(n_points_x)[1] .+ 1) ./ 2
    ys = xs
    for xidx=1:n_points_x
        for yidx=1:n_points_x
            idx = pointidx(xidx-1,yidx-1,(order, order)) + 1
            x = xs[xidx]
            y = ys[yidx]
            points[:,idx] = [x,y]
        end
    end
    return transpose(points), size(points, 2)
end


"""
    VTKPlotter(eq::Equation, scenario::Scenario, grid::Grid,
               filename::String)

Initialize a VTKPlotter for equation `eq` and scenario `scenario`,
defined on `grid`.
Output name is `filename`.
"""
mutable struct VTKPlotter
    eq::Equation
    scenario::Scenario
    grid::AbstractMesh
    filename::String
    collection::WriteVTK.CollectionFile
    plot_counter::Int64
    vtkcells::Array{MeshCell,1}
    vtkpoints::Array{Float64,2}
    cellpoints::Array{Float64,2}

    function VTKPlotter(eq::Equation, scenario::Scenario, grid::AbstractMesh,
                        filename::String)
        collection = paraview_collection(filename) 
        plot_counter = 0

        order = grid.basis.order
        cellpoints, num_points_per_cell = get_plot_points(order)

        vtkcells = Array{MeshCell,1}(undef, length(grid.cells))
        vtkpoints = Array{Float64, 2}(undef, (2, length(grid.cells)*num_points_per_cell))

        for (i,cell) in enumerate(grid.cells)
            offset = cell.center - [cell.size[1]/2, cell.size[2]/2]
            start = (i-1) * num_points_per_cell
            for j=1:num_points_per_cell
                vtkpoints[:, start+j] = offset .+ cellpoints[j,:] .* cell.size
            end
            vtkcells[i] = MeshCell(VTKCellTypes.VTK_LAGRANGE_QUADRILATERAL, Array(start+1:start+num_points_per_cell))
        end


        new(eq, scenario, grid, filename, collection, plot_counter, vtkcells, vtkpoints, cellpoints)
        end
end


function evaluate_dof_points(eq, grid, points, dofidx)
    num_points = size(points, 1)
    eval_data = Array{Float64, 2}(undef, (length(grid.cells)*num_points, 1))
    offset = 0
    for (j, cell) in enumerate(grid.cells)
        for i=1:num_points
            eval_data[offset+i,:] .= evaluate_basis(grid.basis, grid.dofs[:, dofidx, j], points[i,:])
        end
        offset += num_points
    end
    eval_data
end

"""
    plot(plotter::VTKPlotter)

Write output with `plotter` for timestep.
"""
function plot(plotter::VTKPlotter)
    grid = plotter.grid
    eq = plotter.eq

    currentfile = @sprintf("%s_%d", plotter.filename, plotter.plot_counter)
    plotter.plot_counter += 1
    vtkfile = vtk_grid(currentfile, plotter.vtkpoints, plotter.vtkcells)

    for var=1:get_ndofs(eq)
        evaluated_data = evaluate_dof_points(eq, grid, plotter.cellpoints, var)
        vtk_point_data(vtkfile, evaluated_data, string(get_variable_name(eq, var)))
    end
    collection_add_timestep(plotter.collection, vtkfile, grid.time)
end

"""
    save(plotter::VTKPlotter)

Save final output file for `plotter`.
"""
function save(plotter::VTKPlotter)
    vtk_save(plotter.collection)
end