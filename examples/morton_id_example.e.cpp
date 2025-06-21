#include "morton/morton_id.hpp"
#include <iostream>

int main() {
    // Define your Morton ID type (Depth=7, Dimension=2)
    using Morton2D = amr::ndt::morton::morton_id<2u, 2u>;
    
 // 0. Get root
    auto root = Morton2D::zeroth_generation();
    std::cout << "Root ID: " << root << std::endl;


    // 1. Create Morton IDs from coordinates
    Morton2D::coord_array coords1 = {2, 2};
    Morton2D::coord_array coords2 = {0, 0};
    
    auto morton1 = Morton2D::encode(coords1, 2);  // Level 3
    auto morton2 = Morton2D::encode(coords2, 1);  // Level 3
    
    std::cout << "Morton1 ID: " << morton1 << std::endl;
    std::cout << "Morton2 ID: " << morton2 << std::endl;
    
    // 2. Decode Morton IDs back to coordinates
    auto [decoded_coords1, level1] = Morton2D::decode(morton1);
    std::cout << " Morton 1 Decoded: (" << decoded_coords1[0] << ", " 
              << decoded_coords1[1] << ") at level " << (int)level1 << std::endl;
    
    auto [decoded_coords2, level2] = Morton2D::decode(morton2);
    std::cout << " Morton 2 Decoded: (" << decoded_coords2[0] << ", " 
              << decoded_coords2[1] << ") at level " << (int)level2 << std::endl;

    // 3. Get parent
    auto parent1 = Morton2D::parent_of(morton1);
    auto [parent_coords, parent_level] = Morton2D::decode(parent1);
    std::cout << "Morton 1 Parent: (" << parent_coords[0] << ", " 
              << parent_coords[1] << ") at level " << (int)parent_level << std::endl;
    
     // 4. Get zero child (same coordinates, level + 1)
    auto zero_child = Morton2D::child_of(morton2) ;
    auto [child_coords, child_level] = Morton2D::decode(zero_child);
    std::cout << "Zero child of Morton 2 : (" << child_coords[0] << ", " 
              << child_coords[1] << ") at level " << (int)child_level << std::endl;
    
    // 5. Compare Morton IDs
    bool is_less = Morton2D::less(morton1, morton2);
    bool is_equal = Morton2D::equal(morton1, morton2);
    std::cout << "morton1 < morton2: " << is_less << std::endl;
    std::cout << "morton1 == morton2: " << is_equal << std::endl;
    
    // 6. get neighbors
    auto left_neighbor_id = Morton2D::getNeighbor(morton1, Morton2D::direction::left);
    auto [neighbor_coords, neighbor_level] = Morton2D::decode(left_neighbor_id);
    std::cout << "left neighbor of morton 1 : (" << neighbor_coords[0] << ", " 
              << neighbor_coords[1] << ") at level " << (int)neighbor_level << std::endl;
    
    return 0;
}