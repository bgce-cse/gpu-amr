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
    grid::Grid
    filename::String
    collection::WriteVTK.CollectionFile
    plot_counter::Int64

    function VTKPlotter(eq::Equation, scenario::Scenario, grid::Grid,
                        filename::String)
        collection = paraview_collection(filename) 
        plot_counter = 0

        new(eq, scenario, grid, filename, collection, plot_counter)
    end

end

function evaluate_dof(eq, grid, dofidx)
    eval_data = Array{Float64, 2}(undef, (length(grid.cells), 1))
    offset = 1
    for cell in grid.cells
        eval_data[offset] = evaluate_basis(grid.basis, grid.dofs[:, dofidx, cell.dataidx], [0.5, 0.5])
        offset += 1
    end
    eval_data
end

function evaluate_dof_subcell(eq, grid, dofidx)
    order = grid.basis.order
    eval_data = Array{Float64, 2}(undef, (length(grid.cells)*order*order, 1))
    offset = 0
    for cell in grid.cells
        offset_cell = cell.center - [cell.size[1]/2, cell.size[2]/2]
        lin_idx = LinearIndices((order,order))
        for i in CartesianIndices((order,order))
            cellcenter_global = offset_cell + [i[1], i[2]] .* cell.size./order - cell.size./order ./ 2
            cellcenter_reference = localposition(cell, cellcenter_global)
            eval_data[offset+lin_idx[i]] = evaluate_basis(grid.basis, grid.dofs[:, dofidx, cell.dataidx], cellcenter_reference)
        end
        offset += order*order
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
    scenario = plotter.scenario
    order = grid.basis.order
    vtkcells = Array{MeshCell,1}(undef, length(grid.cells)*order*order)
    vtkpoints = Array{Float64, 2}(undef, (2, length(grid.cells)*order*order*4))

    for (i,cell) in enumerate(grid.cells)
        offset = cell.center - [cell.size[1]/2, cell.size[2]/2]
        subgrid = make_mesh(eq, scenario, order, cell.size/order, offset)
        for (j,subcell) in enumerate(subgrid)
            subcell_idx = (i-1)*order*order + j
            start = (subcell_idx-1) * 4 + 1
            vtkpoints[:, start+0] = subcell.center + [ subcell.size[1]/2,  subcell.size[2]/2]
            vtkpoints[:, start+1] = subcell.center + [ subcell.size[1]/2, -subcell.size[2]/2]
            vtkpoints[:, start+2] = subcell.center + [-subcell.size[1]/2, -subcell.size[2]/2]
            vtkpoints[:, start+3] = subcell.center + [-subcell.size[1]/2,  subcell.size[2]/2]
            vtkcells[subcell_idx] = MeshCell(VTKCellTypes.VTK_QUAD, [start, start+1, start+2, start+3])
        end
    end
    currentfile = @sprintf("%s_%d", plotter.filename, plotter.plot_counter)
    plotter.plot_counter += 1
    vtkfile = vtk_grid(currentfile, vtkpoints, vtkcells)

    for var=1:get_ndofs(eq)
        evaluated_data = evaluate_dof_subcell(eq, grid, var)
        vtk_cell_data(vtkfile, evaluated_data, string(get_variable_name(eq, var)))
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