#include <morton/morton2d.hpp>
#include <iostream>
#include <limits> // For debugging purposes
#include <vector>
#include <algorithm> // For std::find

int main() {
    // Set up dummy tree
    uint32_t x, y;
    int temp_level; // Use an intermediate variable to read the input
    uint8_t level;
    std::vector<uint64_t> leaf_vector{};
    leaf_vector.push_back(morton2d::encode2D(0, 0, 1));
    leaf_vector.push_back(morton2d::encode2D(2, 0, 1));
    leaf_vector.push_back(morton2d::encode2D(0, 2, 1));
    leaf_vector.push_back(morton2d::encode2D(2, 2, 2));
    leaf_vector.push_back(morton2d::encode2D(2, 3, 2));
    leaf_vector.push_back(morton2d::encode2D(3, 3, 2));
    leaf_vector.push_back(morton2d::encode2D(3, 2, 2));

    std::cout << "Enter x coordinate: ";
    std::cin >> x;
    std::cout << "Enter y coordinate: ";
    std::cin >> y;
    std::cout << "Enter refinement level: ";
    std::cin >> temp_level;

    // Explicitly cast the input to uint8_t
    level = static_cast<uint8_t>(temp_level);
    std::cout << "Input coordinates: (" << x << ", " << y << ") level: " << static_cast<int>(level) << std::endl;

    uint64_t morton_code = morton2d::encode2D(x, y, level);

    // Check if morton_code exists in leaf_vector
    auto it = std::find(leaf_vector.begin(), leaf_vector.end(), morton_code);
    if (it != leaf_vector.end()) {
        std::cout << "Morton code is in the leaf vector!" << std::endl;
    } else {
        std::cout << "Morton code is NOT in the leaf vector!" << std::endl;
        assert(false);
    }

    //left neighbor
    auto left_morton = morton2d::getNeighbor(morton_code , direction::left);

    if(left_morton.has_value())
    {
        it = std::find(leaf_vector.begin(), leaf_vector.end(), left_morton); // neighbor on the same level exists ?
      if (it != leaf_vector.end()) {
        auto decoded_id = morton2d::decode2D(*left_morton);
        std::cout << "left neighbor found: (" << std::get<0>(decoded_id) << ", " << std::get<1>(decoded_id) << ")"
              << ", level: " << static_cast<int>(std::get<2>(decoded_id)) << std::endl;
        }
        else
        {
            uint64_t left_parent_morton  = morton2d::getParent2D(*left_morton);
            it = std::find(leaf_vector.begin(), leaf_vector.end(), left_parent_morton);
            if (it != leaf_vector.end()) // second case one level -1 neighbor
            {
                auto decoded_id = morton2d::decode2D(left_parent_morton);
                std::cout << "single left neighbor found with level - 1: (" << std::get<0>(decoded_id) << ", " << std::get<1>(decoded_id) << ")"
                << ", level: " << static_cast<int>(std::get<2>(decoded_id)) << std::endl;
            }
            else // third case two left neighbors with level +1
            {
                auto children = morton2d::getChild2D(left_parent_morton);

                auto child0 = morton2d::decode2D(std::get<0>(children));
                auto child2= morton2d::decode2D(std::get<2>(children));

                std::cout << "double left neighbor found with level + 1: (" << std::get<0>(child0) << ", " << std::get<1>(child0) << ")"
                << ", level: " << static_cast<int>(std::get<2>(child0)) << " and "  << std::get<0>(child2) << ", " << std::get<1>(child2) << ")"
                << ", level: " << static_cast<int>(std::get<2>(child2)) <<  std::endl;
            }
            
        }
    }
    else
    {
       std::cout << "no left neighbor exists." << std::endl;
    }

    auto right_morton = morton2d::getNeighbor(morton_code , direction::right);

    if(right_morton.has_value())
    {
        it = std::find(leaf_vector.begin(), leaf_vector.end(), right_morton); // neighbor on the same level exists ?
      if (it != leaf_vector.end()) {
        auto decoded_id = morton2d::decode2D(*right_morton);
        std::cout << "right neighbor found: (" << std::get<0>(decoded_id) << ", " << std::get<1>(decoded_id) << ")"
              << ", level: " << static_cast<int>(std::get<2>(decoded_id)) << std::endl;
        }
        else
        {
            uint64_t right_parent_morton  = morton2d::getParent2D(*right_morton);
            it = std::find(leaf_vector.begin(), leaf_vector.end(), right_parent_morton);
            if (it != leaf_vector.end()) // second case one level -1 neighbor
            {
                auto decoded_id = morton2d::decode2D(right_parent_morton);
                std::cout << "single right neighbor found with level - 1: (" << std::get<0>(decoded_id) << ", " << std::get<1>(decoded_id) << ")"
                << ", level: " << static_cast<int>(std::get<2>(decoded_id)) << std::endl;
            }
            else // third case two left neighbors with level +1
            {
                auto children = morton2d::getChild2D(*right_morton);

                auto child1 = morton2d::decode2D(std::get<1>(children));
                auto child3 = morton2d::decode2D(std::get<3>(children));

                std::cout << "double right neighbor found with level + 1: (" << std::get<0>(child1) << ", " << std::get<1>(child1) << ")"
                << ", level: " << static_cast<int>(std::get<2>(child1)) << " and "  << std::get<0>(child3) << ", " << std::get<1>(child3) << ")"
                << ", level: " << static_cast<int>(std::get<2>(child3)) <<  std::endl;
            }
            
        }
    }
    else
    {
       std::cout << "no right neighbor exists." << std::endl;
    }



    auto top_morton = morton2d::getNeighbor(morton_code, direction::top);

    if (top_morton.has_value()) {
        it = std::find(leaf_vector.begin(), leaf_vector.end(), top_morton); // neighbor on the same level exists?
        if (it != leaf_vector.end()) {
            auto decoded_id = morton2d::decode2D(*top_morton);
            std::cout << "Top neighbor found: (" << std::get<0>(decoded_id) << ", " << std::get<1>(decoded_id) << ")"
                      << ", level: " << static_cast<int>(std::get<2>(decoded_id)) << std::endl;
        } else {
            uint64_t top_parent_morton = morton2d::getParent2D(*top_morton);
            it = std::find(leaf_vector.begin(), leaf_vector.end(), top_parent_morton);
            if (it != leaf_vector.end()) { // Second case: one level -1 neighbor
                auto decoded_id = morton2d::decode2D(top_parent_morton);
                std::cout << "Single top neighbor found with level - 1: (" << std::get<0>(decoded_id) << ", " << std::get<1>(decoded_id) << ")"
                          << ", level: " << static_cast<int>(std::get<2>(decoded_id)) << std::endl;
            } else { // Third case: two top neighbors with level +1
                auto children = morton2d::getChild2D(*top_morton);

                auto child0 = morton2d::decode2D(std::get<0>(children));
                auto child1 = morton2d::decode2D(std::get<1>(children));

                std::cout << "Double top neighbor found with level + 1: (" << std::get<0>(child0) << ", " << std::get<1>(child0) << ")"
                          << ", level: " << static_cast<int>(std::get<2>(child0)) << " and (" << std::get<0>(child1) << ", " << std::get<1>(child1) << ")"
                          << ", level: " << static_cast<int>(std::get<2>(child1)) << std::endl;
            }
        }
    } else {
        std::cout << "No top neighbor exists." << std::endl;
    }

    // Bottom neighbor
    auto bottom_morton = morton2d::getNeighbor(morton_code, direction::bottom);

    if (bottom_morton.has_value()) {
        it = std::find(leaf_vector.begin(), leaf_vector.end(), bottom_morton); // neighbor on the same level exists?
        if (it != leaf_vector.end()) {
            auto decoded_id = morton2d::decode2D(*bottom_morton);
            std::cout << "Bottom neighbor found: (" << std::get<0>(decoded_id) << ", " << std::get<1>(decoded_id) << ")"
                      << ", level: " << static_cast<int>(std::get<2>(decoded_id)) << std::endl;
        } else {
            uint64_t bottom_parent_morton = morton2d::getParent2D(*bottom_morton);
            it = std::find(leaf_vector.begin(), leaf_vector.end(), bottom_parent_morton);
            if (it != leaf_vector.end()) { // Second case: one level -1 neighbor
                auto decoded_id = morton2d::decode2D(bottom_parent_morton);
                std::cout << "Single bottom neighbor found with level - 1: (" << std::get<0>(decoded_id) << ", " << std::get<1>(decoded_id) << ")"
                          << ", level: " << static_cast<int>(std::get<2>(decoded_id)) << std::endl;
            } else { // Third case: two bottom neighbors with level +1
                auto children = morton2d::getChild2D(*bottom_morton);

                auto child2 = morton2d::decode2D(std::get<2>(children));
                auto child3 = morton2d::decode2D(std::get<3>(children));

                std::cout << "Double bottom neighbor found with level + 1: (" << std::get<0>(child2) << ", " << std::get<1>(child2) << ")"
                          << ", level: " << static_cast<int>(std::get<2>(child2)) << " and (" << std::get<0>(child3) << ", " << std::get<1>(child3) << ")"
                          << ", level: " << static_cast<int>(std::get<2>(child3)) << std::endl;
            }
        }
    } else {
        std::cout << "No bottom neighbor exists." << std::endl;
    }

    
    return 0;
}