using DataStructures
using NearestNeighbors

"""
Adaptive Mesh Refinement QuadTree data structure for finite element methods.
Maintains a hierarchical grid with automatic refinement and coarsening based on solution gradients.
"""
mutable struct AMRQuadTree <: AbstractMesh
    root::QuadTreeNode
    max_level::Int
    balance_constraint::Int
    node_counter::Int
    all_nodes::Vector{QuadTreeNode}
    basis::Basis
    size::Array{Float64, 1}
    cells::Vector{QuadTreeNode}
    maxeigenval::Float64
    time::Float64
    ndofs::Int
    eq::Equation
    scenario::Scenario
    current_refinement_level::Int
    dofs::CellArrayView
    flux::CellArrayView

    """
    Construct an AMRQuadTree with initial uniform refinement to specified grid size.
    """
    function AMRQuadTree(config::Configuration, eq::Equation, scenario::Scenario, balance_constraint::Int=1)
        @assert(log(2, (config.grid_elements)) % 1 == 0)
        max_level = config.max_level
        gridsize_1d = config.grid_elements
        gridsize = gridsize_1d^2
        order = config.order
        size = config.physicalsize
        root = QuadTreeNode(0, 0.0, 0.0, size, order, get_ndofs(eq), nothing, 1)
        cells = Vector{QuadTreeNode}(undef, gridsize)
        basis = Basis(order, 2)

        tree = new(root, max_level, balance_constraint, 1, [root], basis, size, cells,
                   -1.0, 0.0, get_ndofs(eq), eq, scenario,
                   log(2, (config.grid_elements)))

        initial_refinement_level = Int(log2(config.grid_elements))
        refine_to_level!(tree, root, initial_refinement_level, false)
        tree.cells = get_leaf_nodes(tree)
        tree.dofs = CellArrayView(tree.cells, :dofs_node)
        tree.flux = CellArrayView(tree.cells, :flux_node)
        update_all_neighbors!(tree)

        return tree
    end
end

"""
Determine which child quadrant a point belongs to based on relative coordinates.
"""
@inline function get_child_index(x_rel::Float64, y_rel::Float64)
    if x_rel < 0.5 && y_rel < 0.5
        return 1
    elseif x_rel >= 0.5 && y_rel < 0.5
        return 2
    elseif x_rel < 0.5 && y_rel >= 0.5
        return 3
    else
        return 4
    end
end

"""
Calculate the coordinates of a child node given parent node and child index.
"""
@inline function get_child_coords(parent::QuadTreeNode, child_idx::Int)
    half_size = parent.size ./ 2
    if child_idx == 1
        return parent.x, parent.y
    elseif child_idx == 2
        return parent.x + half_size[1], parent.y
    elseif child_idx == 3
        return parent.x, parent.y + half_size[2]
    else
        return parent.x + half_size[1], parent.y + half_size[2]
    end
end

"""
Check if two nodes are spatial neighbors (share an edge or corner).
"""
@inline function are_neighbors(node1::QuadTreeNode, node2::QuadTreeNode)
    x1, y1, s1 = node1.x, node1.y, node1.size
    x2, y2, s2 = node2.x, node2.y, node2.size
    x_overlap = !(x1 + s1[1] <= x2 || x2 + s2[1] <= x1)
    y_overlap = !(y1 + s1[2] <= y2 || y2 + s2[2] <= y1)
    edge_x = (x1 + s1[1] ≈ x2 || x2 + s2[1] ≈ x1) && y_overlap
    edge_y = (y1 + s1[2] ≈ y2 || y2 + s2[2] ≈ y1) && x_overlap
    return edge_x || edge_y
end

"""
Determine the direction from node1 to node2 (N, S, E, W) and face index.
"""
@inline function get_direction(node1::QuadTreeNode, node2::QuadTreeNode)
    x1, y1, s1 = node1.x, node1.y, node1.size
    x2, y2, s2 = node2.x, node2.y, node2.size
    center1_x, center1_y = x1 + s1[1]/2, y1 + s1[2]/2
    center2_x, center2_y = x2 + s2[1]/2, y2 + s2[2]/2
    dx = center2_x - center1_x
    dy = center2_y - center1_y
    if abs(dx) > abs(dy)
        return dx > 0 ? (E, 3) : (W, 1)
    else
        return dy > 0 ? (N, 2) : (S, 4)
    end
end

"""
Find all neighbors of a node, including periodic neighbors if applicable.
Updates the node's neighbor lists and face types.
"""
function find_neighbors!(tree::AMRQuadTree, node::QuadTreeNode)
    @inbounds for (i, dir) in enumerate(instances(Face))
        empty!(node.neighbors[dir])
        node.facetypes[i] = boundary
    end

    @inbounds for other_node in tree.all_nodes
        if other_node != node && other_node.is_leaf
            if are_neighbors(node, other_node)
                dir, pos = get_direction(node, other_node)
                node.facetypes[pos] = regular
                push!(node.neighbors[dir], other_node)
            else
                periodic_neighbor_dir = check_periodic_neighbor(tree, node, other_node)
                if periodic_neighbor_dir !== nothing
                    dir, pos = periodic_neighbor_dir
                    push!(node.neighbors[dir], other_node)
                    if is_periodic_boundary(tree.eq, tree.scenario)
                        node.facetypes[pos] = regular
                    else
                        node.facetypes[pos] = boundary
                    end
                end
            end
        end
    end
end

"""
Check if two nodes are periodic neighbors across domain boundaries.
"""
@inline function check_periodic_neighbor(tree::AMRQuadTree, node1::QuadTreeNode, node2::QuadTreeNode)
    domain_size = tree.size
    x1, y1, s1 = node1.x, node1.y, node1.size
    x2, y2, s2 = node2.x, node2.y, node2.size
    if (x1 + s1[1] ≈ domain_size[1]) && (x2 ≈ 0.0)
        y_overlap = !(y1 + s1[2] <= y2 || y2 + s2[2] <= y1)
        if y_overlap return (E, 3) end
    end
    if (x1 ≈ 0.0) && (x2 + s2[1] ≈ domain_size[1])
        y_overlap = !(y1 + s1[2] <= y2 || y2 + s2[2] <= y1)
        if y_overlap return (W, 1) end
    end
    if (y1 + s1[2] ≈ domain_size[2]) && (y2 ≈ 0.0)
        x_overlap = !(x1 + s1[1] <= x2 || x2 + s2[1] <= x1)
        if x_overlap return (N, 2) end
    end
    if (y1 ≈ 0.0) && (y2 + s2[2] ≈ domain_size[2])
        x_overlap = !(x1 + s1[1] <= x2 || x2 + s2[1] <= x1)
        if x_overlap return (S, 4) end
    end
    return nothing
end

"""
Update neighbor relationships for all leaf nodes in the tree.
"""
function update_all_neighbors!(tree::AMRQuadTree)
    @inbounds for node in tree.all_nodes
        if node.is_leaf
            find_neighbors!(tree, node)
        end
    end
    update_tree_views!(tree)
end

"""
Update neighbor relationships for a subset of nodes and their immediate neighbors.
More efficient than updating all neighbors when only local changes occur.
"""
function update_local_neighbors!(tree::AMRQuadTree, nodes_to_update::Vector{QuadTreeNode})
    updated_nodes = Set{QuadTreeNode}()
    @inbounds for node in nodes_to_update
        if node.is_leaf
            find_neighbors!(tree, node)
            push!(updated_nodes, node)
        end
        for dir in instances(Face)
            for neighbor in node.neighbors[dir]
                if neighbor.is_leaf && !(neighbor in updated_nodes)
                    find_neighbors!(tree, neighbor)
                    push!(updated_nodes, neighbor)
                end
            end
        end
    end
    update_tree_views!(tree)
end

"""
Update the tree's cell views after structural changes.
"""
@inline function update_tree_views!(tree::AMRQuadTree)
    tree.cells = get_leaf_nodes(tree)
    tree.dofs = CellArrayView(tree.cells, :dofs_node)
    tree.flux = CellArrayView(tree.cells, :flux_node)
end

"""
Recursively refine a node and all its children to a target refinement level.
"""
function refine_to_level!(tree::AMRQuadTree, node::QuadTreeNode, target_level::Int, can_coarsen_children::Bool = true)
    if node.level >= target_level return end
    node.children = Vector{Union{QuadTreeNode, Nothing}}(undef, 4)
    node.is_leaf = false
    half_size = node.size ./ 2

    @inbounds for i in 1:4
        x, y = get_child_coords(node, i)
        child_center = [x + half_size[1] * 0.5, y + half_size[2] * 0.5]
        tree.node_counter += 1
        idx = compute_morton_index(child_center, tree.max_level)
        child = QuadTreeNode(node.level + 1, x, y, half_size, tree.basis.order,
                             tree.ndofs, node, idx, can_coarsen_children)
        node.children[i] = child
        push!(tree.all_nodes, child)
        if node.level + 1 < target_level
            refine_to_level!(tree, child, target_level, can_coarsen_children)
        end
    end
end

"""
Refine a single node by creating four children, respecting balance constraints.
Returns true if refinement was successful.
"""
function refine_node!(tree::AMRQuadTree, node::QuadTreeNode)
    if !node.is_leaf || node.level >= tree.max_level return false end
    for dir in instances(Face)
        @inbounds for neighbor in node.neighbors[dir]
            if neighbor.level < node.level - tree.balance_constraint
                return false
            end
        end
    end

    node.children = Vector{Union{QuadTreeNode, Nothing}}(undef, 4)
    node.is_leaf = false
    half_size = node.size ./ 2

    @inbounds for i in 1:4
        x, y = get_child_coords(node, i)
        child_center = [x + half_size[1] * 0.5, y + half_size[2] * 0.5]
        tree.node_counter += 1
        idx = compute_morton_index(child_center, tree.max_level)
        child = QuadTreeNode(node.level + 1, x, y, half_size, tree.basis.order, tree.ndofs, node, idx)
        node.children[i] = child
        push!(tree.all_nodes, child)
    end

    update_local_neighbors!(tree, [child for child in node.children if child !== nothing])
    if node.children[1].level > tree.current_refinement_level
        tree.current_refinement_level = node.children[1].level
    end
    interpolate_children(tree.eq, tree.basis, node)
    enforce_balance!(tree)

    return true
end

"""
Coarsen a node by removing its children and making it a leaf again.
Returns true if coarsening was successful.
"""
function coarsen_node!(tree::AMRQuadTree, node::QuadTreeNode)
    if node.is_leaf || !node.can_coarsen return false end
    children_to_remove = QuadTreeNode[]
    for child in node.children
        if child === nothing continue end
        if !child.is_leaf return false end
        push!(children_to_remove, child)
    end
    if isempty(children_to_remove) return false end

    for child in children_to_remove
        for dir in instances(Face)
            for neighbor in child.neighbors[dir]
                if neighbor.level > node.level + tree.balance_constraint
                    return false
                end
            end
        end
    end

    interpolate_parent(tree.eq, tree.basis, node)
    filter!(n -> !(n in children_to_remove), tree.all_nodes)
    fill!(node.children, nothing)
    node.is_leaf = true
    update_all_neighbors!(tree)

    return true
end

"""
Enforce the balance constraint by refining nodes that violate it.
Ensures no node has neighbors that differ by more than the balance constraint levels.
"""
function enforce_balance!(tree::AMRQuadTree)
    changed = true
    while changed
        changed = false
        nodes_copy = copy(tree.all_nodes)
        @inbounds for node in nodes_copy
            if node.is_leaf
                for dir in instances(Face)
                    for neighbor in node.neighbors[dir]
                        if neighbor.level < node.level - tree.balance_constraint
                            if refine_node!(tree, neighbor)
                                changed = true
                            end
                        end
                    end
                end
            end
        end
    end
    update_tree_views!(tree)
end

"""
Get all neighbors of a node in a specific direction.
"""
@inline function get_neighbors(node::QuadTreeNode, direction::Face)
    return node.neighbors[direction]
end

"""
Get all leaf nodes in the tree, sorted by Morton index for consistent ordering.
"""
function get_leaf_nodes(tree::AMRQuadTree)
    leaves = filter(node -> node.is_leaf, tree.all_nodes)
    sorted_leaves = sort(leaves, by = node -> node.id)
    @inbounds for (i, leaf) in enumerate(sorted_leaves)
        leaf.dataidx = i
    end
    return sorted_leaves
end

"""
Iterate over all leaf nodes and return their positions and sizes.
"""
function loop_over_leaves(tree::AMRQuadTree)
    leaf_nodes = get_leaf_nodes(tree)
    positions = []
    @inbounds for node in leaf_nodes
        push!(positions, (node.x, node.y, node.size))
    end
    return positions
end

"""
    amr_update!(tree::AMRQuadTree, eq::Equation, scenario::Scenario, globals::GlobalMatrices; 
                refine_ratio::Float64 = 0.3,     coarsen_ratio::Float64 = 0.25)

Perform adaptive mesh refinement based on acoustic wave solution gradients.
Refines cells with high gradients and coarsens cells with low gradients.
Returns true if any modifications were made.
"""
function amr_update!(tree::AMRQuadTree, eq::Equation, scenario::Scenario, globals::GlobalMatrices; 
                     threshold::Float64 = 0.0)  
    @assert(threshold<=1 && threshold >=0) 
    leaf_nodes = get_leaf_nodes(tree)
    n_leaves = length(leaf_nodes)
    if n_leaves == 0 return false end
    
    # Calculate refinement indicators
    indicators = [calculate_refinement_indicator(tree.basis,cell,globals) for cell in leaf_nodes]
    indicators .= (indicators.-minimum(indicators))/(maximum(indicators)-minimum(indicators))

    average = sum(indicators)/length(indicators)-threshold
    
    
    mesh_changed = false
    
    # Refinement phase
    @inbounds for (i, cell) in enumerate(leaf_nodes)
        if indicators[i] >= average
            if refine_node!(tree, cell)
                mesh_changed = true
            end
        end
    end    

    # Coarsening phase
    @inbounds for (i, cell) in enumerate(leaf_nodes)
        if indicators[i] < average
            if coarsen_node!(tree, cell.parent)
                mesh_changed = true
            end
        end
    end    
    return mesh_changed
end

"""
    calculate_refinement_indicator(cell::QuadTreeNode)

Calculate simple refinement indicator based on DOF gradients in cell.
"""
@inline function calculate_refinement_indicator(basis::Basis,cell::QuadTreeNode,globals::GlobalMatrices)
    faces = globals.project_dofs_to_face
    dofs_N = faces[N] * cell.dofs_node
    dofs_S = faces[S] * cell.dofs_node
    dofs_E = faces[E] * cell.dofs_node
    dofs_W = faces[W] * cell.dofs_node
    invlen = 1/length(dofs_N[:, 3])

    dy = (sum(dofs_N[:, 3] .- dofs_S[:, 3]))*invlen /cell.size[2]
    dx = (sum(dofs_E[:, 3] .- dofs_W[:, 3]))*invlen /cell.size[1]

    grad = norm([dx,dy])

return grad

end


"""
Interpolate solution from parent to children during refinement.
"""
function interpolate_children(eq::Equation, basis::Basis, cell::QuadTreeNode)
    @inbounds for (child_relative_idx, child) in enumerate(cell.children)
        interpolate_children_dofs(eq, cell.dofs_node, child.dofs_node, child, basis, child_relative_idx)
    end
end

"""
Interpolate solution from children to parent during coarsening.
"""
function interpolate_parent(eq::Equation, basis::Basis, cell::QuadTreeNode)
    @inbounds interpolate_parent_dofs(eq, cell, basis)
end

"""
Calculate the relative position of a child within its parent cell.
"""
@inline function relative_child_position(relative_idx, position)
    offset_x = (relative_idx == 1 || relative_idx == 3) ? 0.0 : 0.5
    offset_y = (relative_idx == 1 || relative_idx == 2) ? 0.0 : 0.5
    return position .* 0.5 .+ [offset_x, offset_y]
end

"""
Calculate the relative position within the parent and determine child index.
"""
@inline function relative_parent_position(position)
    ix = position[1] >= 0.5
    iy = position[2] >= 0.5
    child_idx = 1 + ix + 2 * iy
    offset_x = ix ? 0.5 : 0.0
    offset_y = iy ? 0.5 : 0.0
    child_position = 2.0 .* (position .- (offset_x, offset_y))
    return child_position, child_idx
end