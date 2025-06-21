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

    //test offset function
    std::cout << "\n=== OFFSET FUNCTIONS ===\n";
    uint32_t current_offset = Morton2D::offset_of(morton_id);
    std::cout << "Current offset: " << (int)current_offset << std::endl;

    if (current_offset == 0)
    {
        std::cout << "This is a zero sibling, computing other siblings:\n";
        
        // Return all ids of siblings 1, 2 and 3
        auto sibling_1 = Morton2D::offset(morton_id, 1);
        auto sibling_2 = Morton2D::offset(morton_id, 2);
        auto sibling_3 = Morton2D::offset(morton_id, 3);  // Fixed variable name
        
        // Decode and print each sibling
        auto [coords_1, level_1] = Morton2D::decode(sibling_1);
        auto [coords_2, level_2] = Morton2D::decode(sibling_2);
        auto [coords_3, level_3] = Morton2D::decode(sibling_3);
        
        std::cout << "Sibling 1 (offset=1): (" << coords_1[0] << ", " << coords_1[1] 
                  << ") at level " << (int)level_1 << " (ID: " << sibling_1 << ")" << std::endl;
        std::cout << "Sibling 2 (offset=2): (" << coords_2[0] << ", " << coords_2[1] 
                  << ") at level " << (int)level_2 << " (ID: " << sibling_2 << ")" << std::endl;
        std::cout << "Sibling 3 (offset=3): (" << coords_3[0] << ", " << coords_3[1] 
                  << ") at level " << (int)level_3 << " (ID: " << sibling_3 << ")" << std::endl;
        
        // Verify the offset extraction works
        std::cout << "\nVerifying offset extraction:\n";
        std::cout << "Original (offset=0): " << Morton2D::offset_of(morton_id) << std::endl;
        std::cout << "Sibling 1 offset: " << Morton2D::offset_of(sibling_1) << std::endl;
        std::cout << "Sibling 2 offset: " << Morton2D::offset_of(sibling_2) << std::endl;
        std::cout << "Sibling 3 offset: " << Morton2D::offset_of(sibling_3) << std::endl;
    }
    else
    {
        std::cout << "This is not a zero sibling (offset=" << (int)current_offset 
                  << "), cannot compute other siblings from this ID." << std::endl;
        
        // You could compute the zero sibling and then get others:
        auto zero_sibling = morton_id - (current_offset << 6);  // Remove current offset
        std::cout << "Zero sibling would be: " << zero_sibling << std::endl;
    }
    

    
    return 0;
}