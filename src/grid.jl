"""
    @enum FaceType regular=1 boundary=2

Regular faces are faces that have another cell as neighbor.
Boundary faces are faces on the boundary, i.e., where we need
to construct a solution at each timestep.
"""
@enum FaceType regular=1 boundary=2

"""
    struct Cell
    
`Cell` stores all information about grid cells, such as their
center `center`, their size `size`.
It also contains information about its `neighbors`.
In case it is neighbored by a boundary cell, `facetypes` 
indicates this and the neighbor is undefined.
Finally, it stores the `dataidx` which denotes at which
position the data for the cell is stored in the relevant
data arrays.
"""
struct Cell
    center::Array{Float64,1}
    size::Array{Float64,1}
    neighbors::Array{Cell, 1}
    facetypes::Array{FaceType, 1}
    dataidx::Int64
end

"""
    mutable struct Grid

`Grid` stores information about the grid and also information
about the whole simulation.
"""
mutable struct Grid
    basis::Basis
    cells::Array{Cell,1}
    size::Array{Float64, 1}
    dofs::Array{Float64,3}
    flux::Array{Float64,3}
    maxeigenval::Float64
    time::Float64
end

"""
    @enum Face left=1 top=2 right=3 bottom=4

`Face` describes the ordering of our faces. The order is irrelevant
as long as the same order is used everywhere.
Using the wrong order leads to very hard bugs!
"""
@enum Face left=1 top=2 right=3 bottom=4

"""
    get_neighbor(eq, scenario, index, maxindex, offset)

Returns the index of the neighboring cell.
Here, `index` is the 2d-index of the current cell,
`maxindex` is the maximum index,
and `offset` is the difference in index of the neighbor.
As minimum index (0,0) is assumed.
Returns both the index of the neighbor and the type of the face.
It handles both periodic boundary cells (returns correct neighbor and regular
face type) and proper boundaries (returns periodic neighbor and boundary face type).

# Implementation
The type of boundaries depends on the `equation` and the `scenario`.
Users can overwrite the function is_periodic_boundary.
"""
function get_neighbor(eq::Equation, scenario::Scenario, index, maxindex, offset)
    facetype = regular
    i1 = index[1] + offset[1]
    i2 = index[2] + offset[2]
    if i1 < 1
        i1 = maxindex[1]
        facetype = boundary
    elseif i1 > maxindex[1]
        i1 = 1
        facetype = boundary
    end
    if i2 < 1
        i2 = maxindex[2]
        facetype = boundary
    elseif i2 > maxindex[2]
        i2 = 1
        facetype = boundary
    end
    if (is_periodic_boundary(eq, scenario))
        facetype = regular
    end
    CartesianIndex(i1, i2), facetype
end

"""
    make_mesh(eq, scenario, gridsize_1d, cellsize, offset)

Returns all cells of a mesh with number of cel(per dimension) given by 
`gridsize_1d`, size of each cell by `cellsize` and `offset` of grid.
"""
function make_mesh(eq, scenario, gridsize_1d, cellsize, offset)
    gridsize = gridsize_1d^2
    cells = Array{Cell,1}(undef, (gridsize))
    linear_grid_index = LinearIndices((gridsize_1d, gridsize_1d))
    for i in CartesianIndices((gridsize_1d, gridsize_1d))
            center = offset + [i[1], i[2]] .* cellsize - cellsize ./ 2
            neighbors = Array{Cell,1}(undef, 4)
            facetypes = Array{FaceType,1}(undef, 4)
            cells[linear_grid_index[i]] = Cell(center, cellsize, neighbors, facetypes, linear_grid_index[i])
    end
    # init neighbors
    for i in CartesianIndices((gridsize_1d, gridsize_1d))
        # Order: Left, Top, Right, Bottom
        lin_i = linear_grid_index[i]
        offsets = [ (-1, 0), (0, 1), (1, 0), (0, -1) ]
        for j=1:4
            offset = offsets[j]
            neighbor, facetype = get_neighbor(eq, scenario, i, (gridsize_1d, gridsize_1d), offset)
            cells[lin_i].neighbors[j] = cells[linear_grid_index[neighbor]]
            cells[lin_i].facetypes[j] = facetype
        end
    end
    cells
end

"""
    make_grid(eq::Equation, scenario::Scenario, gridsize_1d, size, order)


Returns a grid for equation `eq`, scenario `scenario`, with cells of size `size`
    and number of cells per dimension equals to `gridsize_1d`.
"""
function make_grid(eq::Equation, scenario::Scenario, gridsize_1d, size, order, offset)
    gridsize = gridsize_1d^2
    dofs = Array{Float64,3}(undef, (order * order, get_ndofs(eq), gridsize))#1x3x400
    flux = similar(dofs, order^2 * 2, get_ndofs(eq), gridsize)#2*3*400
    cellsize = size ./ gridsize_1d
    cells = make_mesh(eq, scenario, gridsize_1d, cellsize, offset)
    basis = Basis(order, 2)
    Grid(basis, cells, size, dofs, flux, -1.0, 0.0)
end

"""
    make_grid(config::Configuration, eq::Equation, scenario::Scenario)

Returns a grid for configuration `config`, equation `eq` and scenario `scenario`
"""
function make_grid(config::Configuration, eq::Equation, scenario::Scenario)
    make_grid(eq,
        scenario, 
        config.grid_elements, #n_grid (20)
        config.physicalsize, #domain [grid_size, grid_size] length
        config.order,
        config.offset)
end

"""
    globalposition(cell:Cell, coordinate_reference)
    
Returns the global positon of reference coordinates `coordinate_reference`
for a cell with center `cellcenter` and size `cellsize`.
"""
function globalposition(cell::Cell, coordinate_reference)
    if minimum(coordinate_reference) < 0.0 || maximum(coordinate_reference) > 1.0
        throw(BoundsError())
    end
    cell.size .* coordinate_reference .+ cell.center .- 0.5 .* cell.size #coordinate in the global element
end

"""
    localposition(cell::Cell, coordinate_global)

Opposite of `globalposition`.
Returns the reference (local) positon for global coordinates
`coordinate_global` for a cell with center `cellcenter` and size `cellsize`.
"""
function localposition(cell::Cell, coordinate_global)
    coordinate_reference = 1 ./cell.size .* (coordinate_global .- cell.center + 0.5 .* cell.size)
    if minimum(coordinate_reference) < 0.0 || maximum(coordinate_reference) > 1.0
        throw(BoundsError())
    end
    coordinate_reference
end

"""
    volume(cell::Cell)

Returns the volume of a quad `cell`.
In 2D, it returns the area.
"""
function volume(cell::Cell)
    @assert all(y->y==cell.size[1], cell.size)
    prod(cell.size)
end

"""
    area(cell::Cell)

Returns the area of a quad `cell`.
In 1D, it returns the side-length.
"""
function area(cell::Cell)
    @assert all(y->y==cell.size[1], cell.size)
    area = 1.0
    for i=2:length(cell.size)
        area *= cell.size[i]
    end
    area
end

"""
    inverse_jacobian(cell::Cell)

Returns the inverse Jacobian of a quad cell of size
`cellsize`.
"""
function inverse_jacobian(cell::Cell)
    Diagonal(1 ./ cell.size)
end
