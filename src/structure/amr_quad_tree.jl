using DataStructures
using NearestNeighbors

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
    dofs::CellArrayView
    flux::CellArrayView
    

    
    function AMRQuadTree(config::Configuration, eq::Equation, scenario::Scenario, balance_constraint::Int=1)#move max level and balance constraint into config->yaml
        @assert(log(2,(config.grid_elements))%1 == 0)
        max_level = config.max_level
        gridsize_1d = config.grid_elements
        gridsize = gridsize_1d^2
        order = config.order
        size = config.physicalsize
        root = QuadTreeNode(0, 0.0, 0.0, size, order, get_ndofs(eq), nothing, 1)#TODO size[1] implies only square grids, for now...
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
            get_ndofs(eq))
        
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


# Get child index based on relative position
function get_child_index(x_rel::Float64, y_rel::Float64)
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

# Get child coordinates
function get_child_coords(parent::QuadTreeNode, child_idx::Int)
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

# Check if two nodes are neighbors
function are_neighbors(node1::QuadTreeNode, node2::QuadTreeNode)
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

# Determine direction from node1 to node2
function get_direction(node1::QuadTreeNode, node2::QuadTreeNode)
    x1, y1, s1 = node1.x, node1.y, node1.size
    x2, y2, s2 = node2.x, node2.y, node2.size
    
    center1_x, center1_y = x1 + s1[1]/2, y1 + s1[2]/2
    center2_x, center2_y = x2 + s2[1]/2, y2 + s2[2]/2
    
    dx = center2_x - center1_x
    dy = center2_y - center1_y
    
    # Determine primary direction
    if abs(dx) > abs(dy)
        if dx > 0
            return E
        else
            return W
        end
    else
        if dy > 0
            return N
        else
            return S
        end
    end
end

# Find neighbors of a node
function find_neighbors!(tree::AMRQuadTree, node::QuadTreeNode)
    # Initialize: clear neighbors and assume boundary everywhere
    for dir in instances(Face)
        empty!(node.neighbors[dir])
        node.facetypes[Int(dir)] = boundary
    end

    # Find neighbors among all leaf nodes
    for other_node in tree.all_nodes
        if other_node != node && other_node.is_leaf && are_neighbors(node, other_node)
            dir = get_direction(node, other_node)

            # Found a neighbor in this direction => mark as regular
            node.facetypes[Int(dir)] = regular

            push!(node.neighbors[dir], other_node)
        end
    end
end

# Update all neighbor relationships
function update_all_neighbors!(tree::AMRQuadTree)
    for node in tree.all_nodes
        if node.is_leaf
            find_neighbors!(tree, node)
        end
    end
    update_tree_views!(tree)
end

function update_local_neighbors!(tree::AMRQuadTree, nodes_to_update::Vector{QuadTreeNode})
    updated_nodes = Set{QuadTreeNode}()
    for node in nodes_to_update
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
function update_tree_views!(tree::AMRQuadTree)
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
    
    for i in 1:4
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
        for neighbor in node.neighbors[dir]
            if neighbor.level < node.level - tree.balance_constraint
                return false  # Would violate balance constraint
            end
        end
    end
    
    # Create children
    node.children = Vector{Union{QuadTreeNode, Nothing}}(undef, 4)
    node.is_leaf = false
    half_size = node.size ./ 2
    
    for i in 1:4
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

    
    # Check if we need to refine neighbors to maintain balance
    enforce_balance!(tree)
    
    return true
end

# Coarsen a node (remove its children)
function coarsen_node!(tree::AMRQuadTree, node::QuadTreeNode)
    if node.is_leaf || !node.can_coarsen # Added check for can_coarsen
        return false
    end
    
    # Check if all children are leaves
    for child in node.children
        if child !== nothing && !child.is_leaf
            return false
        end
    end
    
    # Check balance constraint - ensure coarsening won't violate balance
    for child in node.children
        if child !== nothing
            for dir in instances(Face)
                for neighbor in child.neighbors[dir]
                    if neighbor.level > node.level + tree.balance_constraint
                        return false
                    end
                end
            end
        end
    end
    
    # Remove children from all_nodes
    for child in node.children
        if child !== nothing
            filter!(n -> n.id != child.id, tree.all_nodes)
        end
    end
    
    # Remove children
    node.children = Vector{Union{QuadTreeNode, Nothing}}(nothing, 4)
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
        for node in copy(tree.all_nodes)
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
function get_neighbors(node::QuadTreeNode, direction::Face)
    return node.neighbors[direction]
end

# Find node at a specific point
function find_node_at_point(tree::AMRQuadTree, x::Float64, y::Float64)
    current = tree.root
    
    while !current.is_leaf
        # Find which child contains the point
        rel_x = (x - current.x) / current.size[1]
        rel_y = (y - current.y) / current.size[2]
        
        child_idx = get_child_index(rel_x, rel_y)
        
        if current.children[child_idx] === nothing
            break
        end
        
        current = current.children[child_idx]
    end
    
    return current
end

# Get all leaf nodes
function get_leaf_nodes(tree::AMRQuadTree)
    leaves = filter(node -> node.is_leaf, tree.all_nodes)
    sorted_leaves = sort(leaves, by = node -> node.id)
    return sorted_leaves
end


# New function: Loop over leaves and their positions
function loop_over_leaves(tree::AMRQuadTree)
    leaf_nodes = get_leaf_nodes(tree)
    positions = []
    for node in leaf_nodes
        push!(positions, (node.x, node.y, node.size))
    end
    return positions
end

# Print tree structure
function print_tree(tree::AMRQuadTree, node::QuadTreeNode=tree.root, indent::Int=0)
    prefix = "  " ^ indent
    println("$(prefix)Node $(node.id): Level $(node.level), " *
            "Pos ($(node.x), $(node.y)), Size $(node.size), " *
            "Leaf: $(node.is_leaf), Can Coarsen: $(node.can_coarsen)") # Added can_coarsen
    
    if !node.is_leaf
        for (i, child) in enumerate(node.children)
            if child !== nothing
                println("$(prefix)  Child $i:")
                print_tree(tree, child, indent + 2)
            end
        end
    end
end

# Print grid representation with cell boundaries( can be commented out)
function print_grid(tree::AMRQuadTree)
    leaf_nodes = get_leaf_nodes(tree)
    
    if isempty(leaf_nodes)
        println("No leaf nodes to display")
        return
    end
    
    # Find the range of coordinates and levels
    min_x = minimum(node.x for node in leaf_nodes)
    max_x = maximum(node.x + node.size[1] for node in leaf_nodes)
    min_y = minimum(node.y for node in leaf_nodes)
    max_y = maximum(node.y + node.size[2] for node in leaf_nodes)
    
    # Create a reasonable resolution based on the finest level
    max_level = maximum(node.level for node in leaf_nodes)
    resolution = 2^max_level * 16  # 16 chars per finest cell
    
    # Create character grid
    char_grid = fill(' ', resolution, resolution)
    
    # Draw each cell
    for node in leaf_nodes
        # Map to grid coordinates
        start_x = Int(round((node.x - min_x) / (max_x - min_x) * (resolution - 1))) + 1
        start_y = Int(round((node.y - min_y) / (max_y - min_y) * (resolution - 1))) + 1
        end_x = Int(round((node.x + node.size[1] - min_x) / (max_x - min_x) * (resolution - 1))) + 1
        end_y = Int(round((node.y + node.size[2] - min_y) / (max_y - min_y) * (resolution - 1))) + 1
        
        # Clamp to bounds
        start_x = max(1, start_x)
        start_y = max(1, start_y)
        end_x = min(resolution, end_x)
        end_y = min(resolution, end_y)
        
        # Fill interior with level number
        level_char = Char('0' + min(node.level, 9))
        for i in (start_x+1):(end_x-1)
            for j in (start_y+1):(end_y-1)
                char_grid[j, i] = level_char
            end
        end
        
        # Draw boundaries
        # Horizontal lines
        for i in start_x:end_x
            char_grid[start_y, i] = '-'
            char_grid[end_y, i] = '-'
        end
        
        # Vertical lines
        for j in start_y:end_y
            char_grid[j, start_x] = '|'
            char_grid[j, end_x] = '|'
        end
        
        # Corners
        char_grid[start_y, start_x] = '+'
        char_grid[start_y, end_x] = '+'
        char_grid[end_y, start_x] = '+'
        char_grid[end_y, end_x] = '+'
    end
    
    # Print the grid (flip y-axis for proper orientation)
    println("AMR Grid (numbers = refinement levels, |/- = cell boundaries):")
    for j in resolution:-1:1
        for i in 1:resolution
            print(char_grid[j, i])
        end
        println()
    end
end

# Compact grid visualization
function print_compact_grid(tree::AMRQuadTree, size::Int=32)
    leaf_nodes = get_leaf_nodes(tree)
    grid = fill(' ', size, size)
    
    for node in leaf_nodes
        # Map node to grid coordinates
        start_x = Int(floor(node.x * size)) + 1
        start_y = Int(floor(node.y * size)) + 1
        end_x = min(size, Int(ceil((node.x + node.size[1]) * size)))
        end_y = min(size, Int(ceil((node.y + node.size[2]) * size)))
        
        # Clamp to bounds
        start_x = max(1, start_x)
        start_y = max(1, start_y)
        
        # Fill with level indicator
        level_char = Char('0' + min(node.level, 9))
        for i in start_x:end_x
            for j in start_y:end_y
                if i <= size && j <= size
                    grid[j, i] = level_char
                end
            end
        end
    end
    
    # Print grid (flip y-axis)
    println("Compact Grid:")
    for j in size:-1:1
        for i in 1:size
            print(grid[j, i], " ")
        end
        println()
    end
end


