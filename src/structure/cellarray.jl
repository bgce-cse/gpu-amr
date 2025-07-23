
mutable struct QuadTreeNode
    level::Int
    x::Float64
    y::Float64
    center::Array{Float64,1}
    size::Array{Float64,1}
    children::Vector{Union{QuadTreeNode, Nothing}}
    neighbors::Dict{Face, Array{QuadTreeNode,1}}
    parent::Union{QuadTreeNode, Nothing}
    is_leaf::Bool
    id::Int 
    can_coarsen::Bool
    facetypes::Array{FaceType,1}
    dofs_node::Array{Float64,2}
    flux_node::Array{Float64,2}
    dataidx::Int
    
    function QuadTreeNode(level, x, y, size, order, ndofs, parent=nothing, id=0, can_coarsen=true)
        facetypes = Array{FaceType,1}(undef, 4)
        
        children = Vector{Union{QuadTreeNode, Nothing}}(undef, 4)
        fill!(children, nothing)
        
        neighbors = Dict{Face, Vector{QuadTreeNode}}()
        for dir in instances(Face)
            neighbors[dir] = QuadTreeNode[]
        end
        
        node = new(level, x, y, [x + size[1] * 0.5, y + size[2] * 0.5],
                 size, children, neighbors, parent, true, id, can_coarsen, facetypes)
        
        node.dofs_node = Array{Float64,2}(undef, (order^2, ndofs))
        node.flux_node = similar(node.dofs_node, order^2 * 2, ndofs)
        return node
    end
end

"""
Interleaves bits of two 32-bit unsigned integers to produce a Morton index (Z-order curve).
The Morton index provides a spatial ordering that preserves locality in 2D space.
"""
function morton2D(x::UInt32, y::UInt32)::UInt64
    function split_by_1bits(n::UInt32)::UInt64
        n64 = UInt64(n)
        n64 = (n64 | (n64 << 16)) & 0x0000FFFF0000FFFF
        n64 = (n64 | (n64 << 8))  & 0x00FF00FF00FF00FF
        n64 = (n64 | (n64 << 4))  & 0x0F0F0F0F0F0F0F0F
        n64 = (n64 | (n64 << 2))  & 0x3333333333333333
        n64 = (n64 | (n64 << 1))  & 0x5555555555555555
        return n64
    end

    return (split_by_1bits(y) << 1) | split_by_1bits(x)
end

"""
Computes the Morton index for a given center coordinate at the specified maximum level.
Maps continuous coordinates to discrete grid indices and generates the corresponding Morton code.
"""
function compute_morton_index(center::AbstractVector{<:Real}, max_level::Int)
    N = 2^max_level
    dx = 1 / N

    ix_eff = clamp(Int(floor(center[1] / dx)), 0, N - 1)
    iy_eff = clamp(Int(floor(center[2] / dx)), 0, N - 1)

    return morton2D(UInt32(ix_eff), UInt32(iy_eff))
end

"""
Array-like view into QuadTree node data that allows indexing across multiple cells
as if they were a single 3D array. Provides efficient access to either dofs_node
or flux_node fields across all cells in the tree.
"""
struct CellArrayView <: AbstractArray{Float64,3}
    cells_nodes::Vector{QuadTreeNode}
    field::Symbol
end

function Base.size(view::CellArrayView)
    if isempty(view.cells_nodes)
        return (0, 0, 0)
    end
    n_nodes, n_doftypes = size(getfield(view.cells_nodes[1], view.field))
    n_cells = length(view.cells_nodes)
    return (n_nodes, n_doftypes, n_cells)
end

function Base.IndexStyle(::Type{CellArrayView})
    return IndexLinear()
end

"""
Linear indexing for CellArrayView - converts 1D index to (node, doftype, cell) coordinates.
"""
function Base.getindex(view::CellArrayView, i::Int)
    n_nodes, n_doftypes, n_cells = size(view)
    
    cell = div(i - 1, n_nodes * n_doftypes) + 1
    remainder = (i - 1) % (n_nodes * n_doftypes)
    doftype = div(remainder, n_nodes) + 1
    node = remainder % n_nodes + 1
    
    data = getfield(view.cells_nodes[cell], view.field)
    return data[node, doftype]
end

"""
Linear index assignment for CellArrayView.
"""
function Base.setindex!(view::CellArrayView, value, i::Int)
    n_nodes, n_doftypes, n_cells = size(view)
    
    cell = div(i - 1, n_nodes * n_doftypes) + 1
    remainder = (i - 1) % (n_nodes * n_doftypes)
    doftype = div(remainder, n_nodes) + 1
    node = remainder % n_nodes + 1
    
    data = getfield(view.cells_nodes[cell], view.field)
    data[node, doftype] = value
end

"""
Cartesian indexing: view[node, doftype, cell] - direct access to specific element.
"""
function Base.getindex(view::CellArrayView, node::Int, doftype::Int, cell::Int)
    data = getfield(view.cells_nodes[cell], view.field)
    return data[node, doftype]
end

"""
Cartesian index assignment for CellArrayView.
"""
function Base.setindex!(view::CellArrayView, value, node::Int, doftype::Int, cell::Int)
    data = getfield(view.cells_nodes[cell], view.field)
    data[node, doftype] = value
end

"""
Multi-dimensional indexing with support for ranges and colons.
Allows slicing operations like view[:, 1:3, :] similar to regular arrays.
"""
function Base.getindex(view::CellArrayView, nodes::Union{Int, Colon, AbstractRange}, 
                      doftypes::Union{Int, Colon, AbstractRange}, 
                      cells::Union{Int, Colon, AbstractRange})
    n_nodes, n_doftypes, n_cells = size(view)
    node_idx = nodes isa Colon ? (1:n_nodes) : nodes
    dof_idx = doftypes isa Colon ? (1:n_doftypes) : doftypes
    cell_idx = cells isa Colon ? (1:n_cells) : cells
    
    if isa(node_idx, Int) && isa(dof_idx, Int) && isa(cell_idx, Int)
        return view[node_idx, dof_idx, cell_idx]
    end
    
    result_size = (length(node_idx), length(dof_idx), length(cell_idx))
    result = Array{Float64}(undef, result_size)
    
    for (k, cell) in enumerate(cell_idx)
        for (j, dof) in enumerate(dof_idx)
            for (i, node) in enumerate(node_idx)
                result[i, j, k] = view[node, dof, cell]
            end
        end
    end
    
    return result
end

"""
Multi-dimensional assignment with support for ranges, colons, and broadcasting.
"""
function Base.setindex!(view::CellArrayView, value, 
                       nodes::Union{Int, Colon, AbstractRange}, 
                       doftypes::Union{Int, Colon, AbstractRange}, 
                       cells::Union{Int, Colon, AbstractRange})
    n_nodes, n_doftypes, n_cells = size(view)
    node_idx = nodes isa Colon ? (1:n_nodes) : nodes
    dof_idx = doftypes isa Colon ? (1:n_doftypes) : doftypes
    cell_idx = cells isa Colon ? (1:n_cells) : cells
    
    if isa(value, Number)
        for cell in cell_idx
            for dof in dof_idx
                for node in node_idx
                    view[node, dof, cell] = value
                end
            end
        end
    else
        for (k, cell) in enumerate(cell_idx)
            for (j, dof) in enumerate(dof_idx)
                for (i, node) in enumerate(node_idx)
                    view[node, dof, cell] = value[i, j, k]
                end
            end
        end
    end
end

Base.broadcastable(view::CellArrayView) = view

function Base.similar(view::CellArrayView, ::Type{T}, dims::Tuple{Int,Int,Int}) where T
    return Array{T}(undef, dims)
end

function Base.similar(view::CellArrayView, ::Type{T}) where T
    return Array{T}(undef, size(view))
end

function Base.similar(view::CellArrayView)
    return Array{Float64}(undef, size(view))
end

"""
Fills all elements in the view with the specified value.
"""
function Base.fill!(view::CellArrayView, value)
    for cell in view.cells_nodes
        data = getfield(cell, view.field)
        fill!(data, value)
    end
    return view
end

"""
Computes the sum of all elements across all cells in the view.
"""
function Base.sum(view::CellArrayView)
    total = 0.0
    for cell in view.cells_nodes
        data = getfield(cell, view.field)
        total += sum(data)
    end
    return total
end

"""
Finds the maximum value across all cells in the view.
"""
function Base.maximum(view::CellArrayView)
    max_val = -Inf
    for cell in view.cells_nodes
        data = getfield(cell, view.field)
        max_val = max(max_val, maximum(data))
    end
    return max_val
end

"""
Finds the minimum value across all cells in the view.
"""
function Base.minimum(view::CellArrayView)
    min_val = Inf
    for cell in view.cells_nodes
        data = getfield(cell, view.field)
        min_val = min(min_val, minimum(data))
    end
    return min_val
end

"""
Converts the CellArrayView to a regular Julia array by copying all data.
"""
function Base.collect(view::CellArrayView)
    result = Array{Float64}(undef, size(view))
    for (i, val) in enumerate(view)
        result[i] = val
    end
    return result
end