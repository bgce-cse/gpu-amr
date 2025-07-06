include("amr_quad_tree.jl")
include("amr_plotter.jl")
using Random

# Test the AMR Quad Tree
function test_amr_quad_tree()
    # Initialize tree with 4x4 initial grid
    println("Creating AMR Quad Tree with 4x4 initial grid...")
    tree = AMRQuadTree(4, 8, 1)  # 4x4 grid, max level 8, balance constraint 2
    plotter = AMRVTKPlotter("tree")  # will produce tree_0.vtu, tree_1.vtu, ... tree.pvd

    
    # Update initial neighbor relationships
    update_all_neighbors!(tree)
    
    println("Initial tree structure:")
    print_tree(tree)
    println("\nInitial grid:")
    print_grid(tree)
    println("\nNumber of leaf nodes: ", length(get_leaf_nodes(tree)))
    
    # Test coarsening blocking
    println("\nAttempting to coarsen root (should fail as its children are uncoarsenable):")
    if coarsen_node!(tree, tree.root)
        println("ERROR: Coarsened root, but its children should be uncoarsenable!")
    else
        println("Successfully blocked coarsening of root.")
    end

    # Test looping over leaves
    println("\nLooping over leaves and their positions:")
    leaf_positions = loop_over_leaves(tree)
    for (x, y, size) in leaf_positions
        println("  Leaf at: x=$(x), y=$(y), size=$(size)")
    end
    plot(plotter, tree)
    # Interactive loop (optional, for manual testing)
    while true
        println("\n" * "="^50)
        println("AMR Quad Tree Interactive Test")
        println("="^50)
        println("Commands:")
        println("  r x y    - Refine node at position (x, y)")
        println("  c x y    - Coarsen node at position (x, y)")
        println("  rr       - Refine random node")
        println("  rc       - Coarsen random node")
        println("  p        - Print tree structure")
        println("  g        - Print grid with boundaries")
        println("  gc       - Print compact grid")
        println("  s        - Show statistics")
        println("  n x y d  - Show neighbors of node at (x,y) in direction d")
        println("           (directions: N, S, E, W)")
        println("  q        - Quit")
        println("="^50)
        
        print("Enter command: ")
        input = strip(readline())
        
        if input == "q"
            save(plotter)

            break
        elseif input == "p"
            print_tree(tree)
        elseif input == "g"
            print_grid(tree)
        elseif input == "gc"
            print_compact_grid(tree)
        elseif input == "s"
            show_statistics(tree)
        elseif input == "rr"
            refine_random_node!(tree)
            plot(plotter, tree)

        elseif input == "rc"
            coarsen_random_node!(tree)
            plot(plotter, tree)
        elseif startswith(input, "r ")
            parts = split(input)
            if length(parts) == 3
                try
                    x, y = parse(Float64, parts[2]), parse(Float64, parts[3])
                    refine_at_position!(tree, x, y)
                catch e
                    println("Error parsing coordinates: ", e)
                end
            else
                println("Usage: r x y")
            end
            plot(plotter, tree)
        elseif startswith(input, "c ")
            parts = split(input)
            if length(parts) == 3
                try
                    x, y = parse(Float64, parts[2]), parse(Float64, parts[3])
                    coarsen_at_position!(tree, x, y)
                catch e
                    println("Error parsing coordinates: ", e)
                end
            else
                println("Usage: c x y")
            end
            plot(plotter, tree)
        elseif startswith(input, "n ")
            parts = split(input)
            if length(parts) == 4
                try
                    x, y = parse(Float64, parts[2]), parse(Float64, parts[3])
                    dir_str = uppercase(parts[4])
                    show_neighbors_at_position(tree, x, y, dir_str)
                catch e
                    println("Error: ", e)
                end
            else
                println("Usage: n x y direction")
                println("Example: n 0.5 0.5 N")
            end
        else
            println("Unknown command: ", input)
        end
    end
    # After exiting the interactive loop, call:

end

# Refine node at specific position
function refine_at_position!(tree::AMRQuadTree, x::Float64, y::Float64)
    if x < 0 || x > 1 || y < 0 || y > 1
        println("Error: Position must be in [0,1] x [0,1]")
        return
    end
    
    node = find_node_at_point(tree, x, y)
    println("Found node $(node.id) at position ($x, $y)")
    println("Node level: $(node.level), Position: ($(node.x), $(node.y)), Size: $(node.size)")
    
    if refine_node!(tree, node)
        println("Successfully refined node $(node.id)")
        println("New number of leaf nodes: ", length(get_leaf_nodes(tree)))
    else
        println("Could not refine node $(node.id) (may be at max level or violate balance constraint)")
    end
end

# Coarsen node at specific position
function coarsen_at_position!(tree::AMRQuadTree, x::Float64, y::Float64)
    if x < 0 || x > 1 || y < 0 || y > 1
        println("Error: Position must be in [0,1] x [0,1]")
        return
    end
    
    node = find_node_at_point(tree, x, y)
    
    # If it\'s a leaf, try to coarsen its parent
    if node.is_leaf && node.parent !== nothing
        parent = node.parent
        println("Found leaf node $(node.id), attempting to coarsen parent $(parent.id)")
        
        if coarsen_node!(tree, parent)
            println("Successfully coarsened parent node $(parent.id)")
            println("New number of leaf nodes: ", length(get_leaf_nodes(tree)))
        else
            println("Could not coarsen parent node $(parent.id) (children may not be leaves or balance constraint)")
        end
    else
        println("Node $(node.id) is not a leaf or has no parent")
    end
end

# Refine random node
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

# Coarsen random node
function coarsen_random_node!(tree::AMRQuadTree)
    # Get all non-leaf nodes that could potentially be coarsened
    candidates = [node for node in tree.all_nodes if !node.is_leaf]
    
    if isempty(candidates)
        println("No nodes to coarsen")
        return
    end
    
    # Try to coarsen a random node
    max_attempts = 10
    for attempt in 1:max_attempts
        node = rand(candidates)
        if coarsen_node!(tree, node)
            println("Successfully coarsened random node $(node.id) at level $(node.level)")
            println("New number of leaf nodes: ", length(get_leaf_nodes(tree)))
            return
        end
    end
    
    println("Could not coarsen any random node after $max_attempts attempts")
end

# Show statistics
function show_statistics(tree::AMRQuadTree)
    leaf_nodes = get_leaf_nodes(tree)
    level_counts = Dict{Int, Int}()
    
    for node in leaf_nodes
        level_counts[node.level] = get(level_counts, node.level, 0) + 1
    end
    
    println("Tree Statistics:")
    println("  Total nodes: ", length(tree.all_nodes))
    println("  Leaf nodes: ", length(leaf_nodes))
    println("  Max level: ", tree.max_level)
    println("  Balance constraint: ", tree.balance_constraint)
    println("  Level distribution:")
    
    for level in sort(collect(keys(level_counts)))
        println("    Level $level: $(level_counts[level]) nodes")
    end
end

# Show neighbors of node at position
function show_neighbors_at_position(tree::AMRQuadTree, x::Float64, y::Float64, dir_str::String)
    if x < 0 || x > 1 || y < 0 || y > 1
        println("Error: Position must be in [0,1] x [0,1]")
        return
    end
    
    node = find_node_at_point(tree, x, y)
    println("Node $(node.id) at position ($x, $y)")
    
    # Parse direction
    direction = nothing
    if dir_str == "N"
        direction = North
    elseif dir_str == "S"
        direction = South
    elseif dir_str == "E"
        direction = East
    elseif dir_str == "W"
        direction = West
    elseif dir_str == "NE"
        direction = NorthEast
    elseif dir_str == "NW"
        direction = NorthWest
    elseif dir_str == "SE"
        direction = SouthEast
    elseif dir_str == "SW"
        direction = SouthWest
    else
        println("Invalid direction: $dir_str")
        return
    end
    
    neighbors = get_neighbors(node, direction)
    println("Neighbors in direction $dir_str:")
    
    if isempty(neighbors)
        println("  No neighbors")
    else
        for neighbor in neighbors
            println("  Node $(neighbor.id): Level $(neighbor.level), " *
                   "Pos ($(neighbor.x), $(neighbor.y)), Size $(neighbor.size)")
        end
    end
end

# Run the test

    test_amr_quad_tree()

