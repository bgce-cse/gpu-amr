#include "morton/morton_id.hpp"
#include <iostream>
#include <vector>

int main() {
    // Define your Morton ID type (Depth=7, Dimension=2)
    using Morton2D = amr::ndt::morton::morton_id<4u, 2u>;
    
    // Read coordinates and level from user
    uint32_t x, y;
    uint8_t level;
    
    std::cout << "Enter coordinates and level:\n";
    std::cout << "X coordinate: ";
    std::cin >> x;
    std::cout << "Y coordinate: ";
    std::cin >> y;
    std::cout << "Level (0-" << (int)Morton2D::s_depth << "): ";
    int temp_level;
    std::cin >> temp_level;
    level = static_cast<uint8_t>(temp_level);
    
    // Validate input
    if (level > Morton2D::s_depth) {
        std::cerr << "Error: Level too high! Maximum is " << (int)Morton2D::s_depth << std::endl;
        return 1;
    }
    
    Morton2D::coord_array coords = {x, y};
    
    // Create Morton ID from user input
    auto morton_id = Morton2D::encode(coords, level);
    
    std::cout << "\n=== MORTON ID ANALYSIS ===\n";
    std::cout << "Input coordinates: (" << x << ", " << y << ") at level " << (int)level << std::endl;
    std::cout << "Morton ID: " << morton_id << std::endl;
    
    // Decode to verify
    auto [decoded_coords, decoded_level] = Morton2D::decode(morton_id);
    std::cout << "Decoded: (" << decoded_coords[0] << ", " 
              << decoded_coords[1] << ") at level " << (int)decoded_level << std::endl;
    
    // Show hierarchy information
    std::cout << "\n=== HIERARCHY ===\n";
    
    // Root
    auto root = Morton2D::zeroth_generation();
    std::cout << "Root ID: " << root << std::endl;
    
    // Parent (if not root)
   
    auto parent = Morton2D::parent_of(morton_id);
    auto [parent_coords, parent_level] = Morton2D::decode(parent);
    std::cout << "Parent: (" << parent_coords[0] << ", " 
                << parent_coords[1] << ") at level " << (int)parent_level 
                << " (ID: " << parent << ")" << std::endl;

    
    
    // Child (if not at max depth)
    
    auto child = Morton2D::child_of(morton_id);
    auto [child_coords, child_level] = Morton2D::decode(child);
    std::cout << "Zero child: (" << child_coords[0] << ", " 
                << child_coords[1] << ") at level " << (int)child_level 
                << " (ID: " << child << ")" << std::endl;

    
    // All neighbors at same level
    std::cout << "\n=== ALL NEIGHBORS (SAME LEVEL) ===\n";
    
    struct NeighborInfo {
        std::string name;
        Morton2D::direction dir;
    };
    
    std::vector<NeighborInfo> directions = {
        {"Left", Morton2D::direction::left},
        {"Right", Morton2D::direction::right},
        {"Top", Morton2D::direction::top},
        {"Bottom", Morton2D::direction::bottom}
    };
    
    for (const auto& neighbor_info : directions) {
        auto neighbor_id = Morton2D::getNeighbor(morton_id, neighbor_info.dir);
        
        if (neighbor_id != 0) {  // Valid neighbor
            auto [neighbor_coords, neighbor_level] = Morton2D::decode(neighbor_id);
            std::cout << neighbor_info.name << " neighbor: (" 
                      << neighbor_coords[0] << ", " << neighbor_coords[1] 
                      << ") at level " << (int)neighbor_level 
                      << " (ID: " << neighbor_id << ")" << std::endl;
        } else {
            std::cout << neighbor_info.name << " neighbor: Out of bounds" << std::endl;
        }
    }
    
    // Comparison with root
    std::cout << "\n=== COMPARISONS ===\n";
    bool is_less_than_root = Morton2D::less(morton_id, root);
    bool is_equal_to_root = Morton2D::equal(morton_id, root);
    std::cout << "ID < Root: " << is_less_than_root << std::endl;
    std::cout << "ID == Root: " << is_equal_to_root << std::endl;
    
    // Grid information
    std::cout << "\n=== GRID INFORMATION ===\n";
    uint32_t grid_size = 1u << Morton2D::s_depth;
    uint32_t cell_size = 1u << (Morton2D::s_depth - level);
    std::cout << "Total grid size: " << grid_size << "x" << grid_size << std::endl;
    std::cout << "Cell size at level " << (int)level << ": " 
              << cell_size << "x" << cell_size << std::endl;
    std::cout << "Cells per dimension at this level: " 
              << (grid_size / cell_size) << std::endl;
    
    return 0;
}