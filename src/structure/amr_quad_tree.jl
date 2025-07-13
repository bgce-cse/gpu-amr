using DataStructures
using NearestNeighbors

directionidx = Dict(W => 1, N=>2, E=>3, S=>4)

# AMR Quad Tree structure
mutable struct AMRQuadTree <: AbstractMesh
    root::QuadTreeNode
    max_level::Int
    balance_constraint::Int  # maximum level difference between neighbors
    node_counter::Int
    all_nodes::Vector{QuadTreeNode}  # Keep track of all nodes for easier access
    basis::Basis
    size::Array{Float64,1}
    cells::Vector{QuadTreeNode}
    maxeigenval::Float64
    time::Float64
    ndofs::Int
    eq::Equation
    scenario::Scenario
    current_refinement_level::Int
    dofs::CellArrayView
    flux::CellArrayView
    
    function AMRQuadTree(config::Configuration, eq::Equation, scenario::Scenario, balance_constraint::Int=1)
        @assert(log(2,(config.grid_elements))%1 == 0)
        max_level = config.max_level
        gridsize_1d = config.grid_elements
        gridsize = gridsize_1d^2
        order = config.order
        size = config.physicalsize
        root = QuadTreeNode(0, 0.0, 0.0, size, order, get_ndofs(eq), nothing, 1)
        cells = Vector{QuadTreeNode}(undef, (gridsize))
        basis = Basis(order, 2)

        tree = new(root, 
            max_level, 
            balance_constraint, 
            1, 
            [root], 
            basis,
            size,
            cells,
            -1.0,
            0.0,
            get_ndofs(eq),
            eq,
            scenario,
            log(2,(config.grid_elements)))
        
        # Create initial n×n grid with uncoarsenable children
        initial_refinement_level = Int(log2(config.grid_elements))
        refine_to_level!(tree, root, initial_refinement_level, false) # Mark initial grid as uncoarsenable
        tree.cells = get_leaf_nodes(tree)
        tree.dofs = CellArrayView(tree.cells, :dofs_node)
        tree.flux = CellArrayView(tree.cells, :flux_node)
        update_all_neighbors!(tree)
        
        return tree
    end
end

# Get child index based on relative position - optimized with @inline
@inline function get_child_index(x_rel::Float64, y_rel::Float64)
    if x_rel < 0.5 && y_rel < 0.5
        return 1  # SW
    elseif x_rel >= 0.5 && y_rel < 0.5
        return 2  # SE
    elseif x_rel < 0.5 && y_rel >= 0.5
        return 3  # NW
    else
        return 4  # NE
    end
end

# Get child coordinates - optimized with @inline
@inline function get_child_coords(parent::QuadTreeNode, child_idx::Int)
    half_size = parent.size ./ 2
    if child_idx == 1  # SW
        return parent.x, parent.y
    elseif child_idx == 2  # SE
        return parent.x + half_size[1], parent.y
    elseif child_idx == 3  # NW
        return parent.x, parent.y + half_size[2]
    else  # NE
        return parent.x + half_size[1], parent.y + half_size[2]
    end
end

# Check if two nodes are neighbors - optimized with @inline
@inline function are_neighbors(node1::QuadTreeNode, node2::QuadTreeNode)
    # Check if nodes share an edge or corner
    x1, y1, s1 = node1.x, node1.y, node1.size
    x2, y2, s2 = node2.x, node2.y, node2.size
    
    # Check for overlap in x and y ranges
    x_overlap = !(x1 + s1[1] <= x2 || x2 + s2[1] <= x1)
    y_overlap = !(y1 + s1[2] <= y2 || y2 + s2[2] <= y1)
    
    # Check if they share an edge
    edge_x = (x1 + s1[1] ≈ x2 || x2 + s2[1] ≈ x1) && y_overlap
    edge_y = (y1 + s1[2] ≈ y2 || y2 + s2[2] ≈ y1) && x_overlap
    
    return edge_x || edge_y
end

# Determine direction from node1 to node2 - optimized with @inline
@inline function get_direction(node1::QuadTreeNode, node2::QuadTreeNode)
    x1, y1, s1 = node1.x, node1.y, node1.size
    x2, y2, s2 = node2.x, node2.y, node2.size
    
    center1_x, center1_y = x1 + s1[1]/2, y1 + s1[2]/2
    center2_x, center2_y = x2 + s2[1]/2, y2 + s2[2]/2
    
    dx = center2_x - center1_x
    dy = center2_y - center1_y
    
    # Determine primary direction
    if abs(dx) > abs(dy)
        if dx > 0
            return E, 3
        else
            return W, 1
        end
    else
        if dy > 0
            return N, 2
        else
            return S,4
        end
    end
end

# Modified find_neighbors! function with periodic boundary conditions
function find_neighbors!(tree::AMRQuadTree, node::QuadTreeNode)
    # Initialize: clear neighbors and assume boundary everywhere
    @inbounds for (i, dir) in enumerate(instances(Face))
        empty!(node.neighbors[dir])
        node.facetypes[i] = boundary
    end

    # Find neighbors among all leaf nodes (including periodic)
    @inbounds for other_node in tree.all_nodes
        if other_node != node && other_node.is_leaf
            if are_neighbors(node, other_node)
                # Regular neighbor
                dir, pos = get_direction(node, other_node)
                # Found a regular neighbor => mark as regular
                node.facetypes[pos] = regular
                push!(node.neighbors[dir], other_node)
            else
                # Always check for periodic neighbors
                periodic_neighbor_dir = check_periodic_neighbor(tree, node, other_node)
                if periodic_neighbor_dir !== nothing 
                    dir, pos = periodic_neighbor_dir
                    # Found a periodic neighbor => add the neighbor
                    push!(node.neighbors[dir], other_node)
                    # Set facetype based on boundary condition setting
                    if is_periodic_boundary(tree.eq, tree.scenario)
                        node.facetypes[pos] = regular  # Periodic boundaries are regular
                    else
                        node.facetypes[pos] = boundary  # Non-periodic boundaries stay boundary
                    end
                end
            end
        end
    end
end

# New function to check if two nodes are periodic neighbors - optimized with @inline
@inline function check_periodic_neighbor(tree::AMRQuadTree, node1::QuadTreeNode, node2::QuadTreeNode)
    domain_size = tree.size
    x1, y1, s1 = node1.x, node1.y, node1.size
    x2, y2, s2 = node2.x, node2.y, node2.size
    
    # Check East-West periodic boundary
    # Node1 on east boundary, node2 on west boundary
    if (x1 + s1[1] ≈ domain_size[1]) && (x2 ≈ 0.0)
        # Check if they align in y-direction
        y_overlap = !(y1 + s1[2] <= y2 || y2 + s2[2] <= y1)
        if y_overlap
            return (E, 3)
        end
    end
    
    # Node1 on west boundary, node2 on east boundary
    if (x1 ≈ 0.0) && (x2 + s2[1] ≈ domain_size[1])
        y_overlap = !(y1 + s1[2] <= y2 || y2 + s2[2] <= y1)
        if y_overlap
            return (W, 1)
        end
    end
    
    # Check North-South periodic boundary
    # Node1 on north boundary, node2 on south boundary
    if (y1 + s1[2] ≈ domain_size[2]) && (y2 ≈ 0.0)
        # Check if they align in x-direction
        x_overlap = !(x1 + s1[1] <= x2 || x2 + s2[1] <= x1)
        if x_overlap
            return (N, 2)
        end
    end
    
    # Node1 on south boundary, node2 on north boundary
    if (y1 ≈ 0.0) && (y2 + s2[2] ≈ domain_size[2])
        x_overlap = !(x1 + s1[1] <= x2 || x2 + s2[1] <= x1)
        if x_overlap
            return (S, 4)
        end
    end
    
    return nothing
end

# # Alternative: More robust version with tolerance
# function check_periodic_neighbor_robust(tree::AMRQuadTree, node1::QuadTreeNode, node2::QuadTreeNode)
#     domain_size = tree.size
#     x1, y1, s1 = node1.x, node1.y, node1.size
#     x2, y2, s2 = node2.x, node2.y, node2.size
    
#     # Tolerance for floating point comparison
#     tol = 1e-10
    
#     # Check East-West periodic boundary
#     if (abs(x1 + s1[1] - domain_size[1]) < tol) && (abs(x2) < tol)
#         # Check y-overlap
#         if max(y1, y2) < min(y1 + s1[2], y2 + s2[2]) + tol
#             return (E, 3)
#         end
#     end
    
#     if (abs(x1) < tol) && (abs(x2 + s2[1] - domain_size[1]) < tol)
#         if max(y1, y2) < min(y1 + s1[2], y2 + s2[2]) + tol
#             return (W, 1)
#         end
#     end
    
#     # Check North-South periodic boundary
#     if (abs(y1 + s1[2] - domain_size[2]) < tol) && (abs(y2) < tol)
#         # Check x-overlap
#         if max(x1, x2) < min(x1 + s1[1], x2 + s2[1]) + tol
#             return (N, 2)
#         end
#     end
    
#     if (abs(y1) < tol) && (abs(y2 + s2[2] - domain_size[2]) < tol)
#         if max(x1, x2) < min(x1 + s1[1], x2 + s2[1]) + tol
#             return (S, 4)
#         end
#     end
    
#     return nothing
# end

# Update all neighbor relationships
function update_all_neighbors!(tree::AMRQuadTree)
    @inbounds for node in tree.all_nodes
        if node.is_leaf
            find_neighbors!(tree, node)
        end
    end
    update_tree_views!(tree)
end

function update_local_neighbors!(tree::AMRQuadTree, nodes_to_update::Vector{QuadTreeNode})
    updated_nodes = Set{QuadTreeNode}()
    @inbounds for node in nodes_to_update
        if node.is_leaf
            find_neighbors!(tree, node)
            push!(updated_nodes, node)
        end
        # Update immediate neighbors as well
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

# Update the mesh 
@inline function update_tree_views!(tree::AMRQuadTree)
    tree.cells = get_leaf_nodes(tree)
    tree.dofs = CellArrayView(tree.cells, :dofs_node)
    tree.flux = CellArrayView(tree.cells, :flux_node)
end

# Refine a node to a specific level
function refine_to_level!(tree::AMRQuadTree, node::QuadTreeNode, target_level::Int, can_coarsen_children::Bool=true)
    if node.level >= target_level
        return
    end
    
    # Create children
    node.children = Vector{Union{QuadTreeNode, Nothing}}(undef, 4)
    node.is_leaf = false
    half_size = node.size ./ 2
    
    @inbounds for i in 1:4
        x, y = get_child_coords(node, i)
        child_center = [x + half_size[1] * 0.5, y + half_size[2] * 0.5]
        tree.node_counter += 1
        idx = compute_morton_index(child_center, tree.max_level)
        child = QuadTreeNode(node.level + 1, x, y, half_size, tree.basis.order, tree.ndofs, node, idx, can_coarsen_children)
        node.children[i] = child
        push!(tree.all_nodes, child)
        
        # Recursively refine if needed
        if node.level + 1 < target_level
            refine_to_level!(tree, child, target_level, can_coarsen_children)
        end
    end
end

# Refine a single node
function refine_node!(tree::AMRQuadTree, node::QuadTreeNode)
    if !node.is_leaf || node.level >= tree.max_level
        return false
    end
    
    # Check balance constraint
    for dir in instances(Face)
        @inbounds for neighbor in node.neighbors[dir]
            if neighbor.level < node.level - tree.balance_constraint
                return false  # Would violate balance constraint
            end
        end
    end
    
    # Create children
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
    
    # Update neighbor relationships
    update_local_neighbors!(tree, [child for child in node.children if child !== nothing])

    # Update the current refinement level
    if node.children[1].level > tree.current_refinement_level
        tree.current_refinement_level = node.children[1].level
    end
        
    interpolate_children(tree.eq, tree.basis, node)
    # Check if we need to refine neighbors to maintain balance
    enforce_balance!(tree)
    
    return true
end


# Fixed coarsen_node! function
function coarsen_node!(tree::AMRQuadTree, node::QuadTreeNode)
    if node.is_leaf || !node.can_coarsen
        return false
    end
    
    # Check if all children are leaves and exist
    children_to_remove = QuadTreeNode[]
    for child in node.children
        if child === nothing
            continue
        end
        if !child.is_leaf
            return false  # Cannot coarsen if any child is not a leaf
        end
        push!(children_to_remove, child)
    end
    
    # If no children to remove, nothing to do
    if isempty(children_to_remove)
        return false
    end
    
    # Check balance constraint - ensure coarsening won't violate balance
    for child in children_to_remove
        for dir in instances(Face)
            for neighbor in child.neighbors[dir]
                if neighbor.level > node.level + tree.balance_constraint
                    return false
                end
            end
        end
    end
    
    # Interpolate parent data from children before removing them
    interpolate_parent(tree.eq, tree.basis, node)
    
    # Remove children from all_nodes using object identity
    filter!(n -> !(n in children_to_remove), tree.all_nodes)
    
    # FIXED: Properly clear children array
    fill!(node.children, nothing)
    node.is_leaf = true
    
    # Update neighbor relationships
    update_all_neighbors!(tree)
    
    return true
end



# Enforce balance constraint
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

# Get neighbors of a node in a specific direction
@inline function get_neighbors(node::QuadTreeNode, direction::Face)
    return node.neighbors[direction]
end

# # Find node at a specific point
# function find_node_at_point(tree::AMRQuadTree, x::Float64, y::Float64)
#     current = tree.root
    
#     while !current.is_leaf
#         # Find which child contains the point
#         rel_x = (x - current.x) / current.size[1]
#         rel_y = (y - current.y) / current.size[2]
        
#         child_idx = get_child_index(rel_x, rel_y)
        
#         if current.children[child_idx] === nothing
#             break
#         end
        
#         current = current.children[child_idx]
#     end
    
#     return current
# end

# Get all leaf nodes
function get_leaf_nodes(tree::AMRQuadTree)
    leaves = filter(node -> node.is_leaf, tree.all_nodes)
    sorted_leaves = sort(leaves, by = node -> node.id)
    @inbounds for (i, leaf) in enumerate(sorted_leaves)
        leaf.dataidx = i
    end
    return sorted_leaves
end

# New function: Loop over leaves and their positions
function loop_over_leaves(tree::AMRQuadTree)
    leaf_nodes = get_leaf_nodes(tree)
    positions = []
    @inbounds for node in leaf_nodes
        push!(positions, (node.x, node.y, node.size))
    end
    return positions
end

function amr_update!(tree::AMRQuadTree, eq::Equation, scenario::Scenario)
    modified_any = false
    cells_to_refine = QuadTreeNode[]
    cells_to_coarsen = QuadTreeNode[]
    
    @inbounds for cell in tree.cells
        if evaluate_gradient_amr(cell,tree)
            push!(cells_to_refine, cell)
        else
            if cell.parent !== nothing
                push!(cells_to_coarsen, cell.parent)
            end
        end
    end

    for cell in cells_to_refine
        modified_any |= refine_node!(tree, cell)
    end
    for parent in unique(cells_to_coarsen)
        modified_any |= coarsen_node!(tree, parent)
    end

    update_tree_views!(tree)
    return modified_any
end

@inline function evaluate_gradient_amr(
    cell::QuadTreeNode,
    tree::AMRQuadTree,
    threshold_factor::Float64 = 1.2
)
    if !cell.is_leaf || cell.level >= tree.max_level
        return false
    end

    # Compute average absolute value across all entries of dofs_node
    avg_cell_dof = sum(abs, cell.dofs_node) / length(cell.dofs_node)

    if avg_cell_dof < 1e-9
        return false
    end

    total_jump_squared = 0.0

    # Sum the squared L2 norm of the jump with each neighbor
    for dir in instances(Face)
        for neighbor in get_neighbors(cell, dir)

            jump = cell.dofs_node .- neighbor.dofs_node

            total_jump_squared += sum(abs2, jump)
        end
    end

    indicator = sqrt(total_jump_squared)

    # Refine if the indicator exceeds the scaled threshold
    return indicator > threshold_factor * avg_cell_dof
end

function interpolate_children(eq::Equation,basis::Basis, cell::QuadTreeNode)
    @inbounds for (child_relative_idx, child) in enumerate(cell.children)
        interpolate_children_dofs(eq, cell.dofs_node, child.dofs_node, child, basis, child_relative_idx)
    end
end

function interpolate_parent(eq::Equation,basis::Basis, cell::QuadTreeNode)
    @inbounds interpolate_parent_dofs(eq, cell, basis)
end

@inline function relative_child_position(relative_idx, position)
    offset_x = (relative_idx ==  1 || relative_idx == 3) ? 0.0 : 0.5
    offset_y = (relative_idx ==  1 || relative_idx == 2) ? 0.0 : 0.5
    return position.*0.5 .+ [offset_x, offset_y]
end

@inline function relative_parent_position(position)
    offset_x = (position[1] < 0.5) ? 0.0 : -1
    offset_y = (position[2] < 0.5) ? 0.0 : -1
    child_idx = 1 + 2*(offset_x == -1) + 1*(offset_y == -1)
    return position.*2.0 .+ [offset_x, offset_y], child_idx
end