
# Face enumeration for neighbor finding


# Node structure for the quad tree
mutable struct QuadTreeNode
    level::Int
    x::Float64  # x coordinate of bottom-left corner TODO cahnge into the center
    y::Float64  # y coordinate of bottom-left corner
    center::Array{Float64,1}
    size::Array{Float64,1}  # size of the cell
    children::Vector{Union{QuadTreeNode, Nothing}}  # 4 children (SW, SE, NW, NE)
    neighbors::Dict{Face, Array{QuadTreeNode,1}}  # neighbors in each direction
    parent::Union{QuadTreeNode, Nothing}
    is_leaf::Bool
    id::Int 
    can_coarsen::Bool # New field: indicates if this node can be coarsened
    facetypes::Array{FaceType,1}
    dofs_node::Array{Float64,2}
    flux_node::Array{Float64,2}
    dataidx::Int
    # dofs_gradient::similar{dofs_node}
    
    
    function QuadTreeNode(level, x, y, size, order, ndofs, parent=nothing, id=0, can_coarsen=true)
        facetypes = Array{FaceType,1}(undef, 4)
        
        # FIXED: Properly initialize children vector
        children = Vector{Union{QuadTreeNode, Nothing}}(undef, 4)
        fill!(children, nothing)
        
        # FIXED: Initialize neighbors dictionary with empty vectors
        neighbors = Dict{Face, Vector{QuadTreeNode}}()
        for dir in instances(Face)
            neighbors[dir] = QuadTreeNode[]  # Empty vector, not pre-allocated
        end
        
        node = new(level, x, y, [x + size[1] * 0.5, y + size[2] * 0.5],
                 size, children, neighbors, parent, true, id, can_coarsen, facetypes)
        
        node.dofs_node = Array{Float64,2}(undef, (order^2, ndofs))
        node.flux_node = similar(node.dofs_node, order^2 * 2, ndofs)
        return node
    end
end

"""
Interleaves bits of two 32-bit unsigned integers to produce a Morton index.
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

function compute_morton_index(center::AbstractVector{<:Real}, max_level::Int)
    N = 2^max_level
    dx = 1 / N

    ix_eff = clamp(Int(floor(center[1] / dx)), 0, N - 1)
    iy_eff = clamp(Int(floor(center[2] / dx)), 0, N - 1)

    return morton2D(UInt32(ix_eff), UInt32(iy_eff))
end


# Simplified CellArrayView that works reliably with broadcasting
struct CellArrayView <: AbstractArray{Float64,3}
    cells_nodes::Vector{QuadTreeNode}
    field::Symbol  # :dofs_node or :flux_node
end

# Required AbstractArray interface
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

# Linear indexing
function Base.getindex(view::CellArrayView, i::Int)
    n_nodes, n_doftypes, n_cells = size(view)
    
    # Convert linear index to (node, doftype, cell)
    cell = div(i - 1, n_nodes * n_doftypes) + 1
    remainder = (i - 1) % (n_nodes * n_doftypes)
    doftype = div(remainder, n_nodes) + 1
    node = remainder % n_nodes + 1
    
    data = getfield(view.cells_nodes[cell], view.field)
    return data[node, doftype]
end

function Base.setindex!(view::CellArrayView, value, i::Int)
    n_nodes, n_doftypes, n_cells = size(view)
    
    # Convert linear index to (node, doftype, cell)
    cell = div(i - 1, n_nodes * n_doftypes) + 1
    remainder = (i - 1) % (n_nodes * n_doftypes)
    doftype = div(remainder, n_nodes) + 1
    node = remainder % n_nodes + 1
    
    data = getfield(view.cells_nodes[cell], view.field)
    data[node, doftype] = value
end

# Cartesian indexing: view[node, doftype, cell]
function Base.getindex(view::CellArrayView, node::Int, doftype::Int, cell::Int)
    data = getfield(view.cells_nodes[cell], view.field)
    return data[node, doftype]
end

function Base.setindex!(view::CellArrayView, value, node::Int, doftype::Int, cell::Int)
    data = getfield(view.cells_nodes[cell], view.field)
    data[node, doftype] = value
end

# Support for ranges and colons
function Base.getindex(view::CellArrayView, nodes::Union{Int, Colon, AbstractRange}, 
                      doftypes::Union{Int, Colon, AbstractRange}, 
                      cells::Union{Int, Colon, AbstractRange})
    # Convert to actual indices
    n_nodes, n_doftypes, n_cells = size(view)
    node_idx = nodes isa Colon ? (1:n_nodes) : nodes
    dof_idx = doftypes isa Colon ? (1:n_doftypes) : doftypes
    cell_idx = cells isa Colon ? (1:n_cells) : cells
    
    # Handle scalar indexing
    if isa(node_idx, Int) && isa(dof_idx, Int) && isa(cell_idx, Int)
        return view[node_idx, dof_idx, cell_idx]
    end
    
    # Create result array
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

function Base.setindex!(view::CellArrayView, value, 
                       nodes::Union{Int, Colon, AbstractRange}, 
                       doftypes::Union{Int, Colon, AbstractRange}, 
                       cells::Union{Int, Colon, AbstractRange})
    # Convert to actual indices
    n_nodes, n_doftypes, n_cells = size(view)
    node_idx = nodes isa Colon ? (1:n_nodes) : nodes
    dof_idx = doftypes isa Colon ? (1:n_doftypes) : doftypes
    cell_idx = cells isa Colon ? (1:n_cells) : cells
    
    # Handle broadcasting
    if isa(value, Number)
        for cell in cell_idx
            for dof in dof_idx
                for node in node_idx
                    view[node, dof, cell] = value
                end
            end
        end
    else
        # Handle array assignment
        for (k, cell) in enumerate(cell_idx)
            for (j, dof) in enumerate(dof_idx)
                for (i, node) in enumerate(node_idx)
                    view[node, dof, cell] = value[i, j, k]
                end
            end
        end
    end
end

# Simple broadcasting support - use default Julia broadcasting
Base.broadcastable(view::CellArrayView) = view

# Additional useful methods for array-like behavior
function Base.similar(view::CellArrayView, ::Type{T}, dims::Tuple{Int,Int,Int}) where T
    return Array{T}(undef, dims)
end

function Base.similar(view::CellArrayView, ::Type{T}) where T
    return Array{T}(undef, size(view))
end

function Base.similar(view::CellArrayView)
    return Array{Float64}(undef, size(view))
end

# Common array operations
function Base.fill!(view::CellArrayView, value)
    for cell in view.cells_nodes
        data = getfield(cell, view.field)
        fill!(data, value)
    end
    return view
end

# Reduction operations
function Base.sum(view::CellArrayView)
    total = 0.0
    for cell in view.cells_nodes
        data = getfield(cell, view.field)
        total += sum(data)
    end
    return total
end

function Base.maximum(view::CellArrayView)
    max_val = -Inf
    for cell in view.cells_nodes
        data = getfield(cell, view.field)
        max_val = max(max_val, maximum(data))
    end
    return max_val
end

function Base.minimum(view::CellArrayView)
    min_val = Inf
    for cell in view.cells_nodes
        data = getfield(cell, view.field)
        min_val = min(min_val, minimum(data))
    end
    return min_val
end

# Norm operations
function norm(view::CellArrayView, p::Real=2)
    if p == 2
        total = 0.0
        for cell in view.cells_nodes
            data = getfield(cell, view.field)
            total += sum(abs2, data)
        end
        return sqrt(total)
    else
        return norm(collect(view), p)
    end
end

# Convert to regular array when needed
function Base.collect(view::CellArrayView)
    result = Array{Float64}(undef, size(view))
    for (i, val) in enumerate(view)
        result[i] = val
    end
    return result
end

# Efficient conversion to Array{Float64,3}
function to_array(view::CellArrayView)
    n_nodes, n_doftypes, n_cells = size(view)
    result = Array{Float64,3}(undef, n_nodes, n_doftypes, n_cells)
    
    for (i, cell) in enumerate(view.cells_nodes)
        data = getfield(cell, view.field)
        result[:, :, i] = data
    end
    
    return result
end

# Update from Array{Float64,3}
function from_array!(view::CellArrayView, arr::Array{Float64,3})
    @assert size(arr) == size(view) "Array dimensions must match"
    
    for (i, cell) in enumerate(view.cells_nodes)
        data = getfield(cell, view.field)
        data[:, :] = arr[:, :, i]
    end
    
    return view
end

# Convenient functions for common operations
function add_scalar!(view::CellArrayView, value::Number)
    for cell in view.cells_nodes
        data = getfield(cell, view.field)
        data .+= value
    end
    return view
end

function multiply_scalar!(view::CellArrayView, value::Number)
    for cell in view.cells_nodes
        data = getfield(cell, view.field)
        data .*= value
    end
    return view
end