#include <morton/morton2d.hpp>
#include <iostream>
#include <limits> // For debugging purposes

int main() {
    morton2d::initialize(2);
    // Take user input for 2D coordinates
    uint32_t x, y;
    int temp_level; // Use an intermediate variable to read the input
    uint8_t level;

    std::cout << "Enter x coordinate: ";
    std::cin >> x;
    std::cout << "Enter y coordinate: ";
    std::cin >> y;
    std::cout << "Enter refinement level: ";
    std::cin >> temp_level;

    // Explicitly cast the input to uint8_t
    level = static_cast<uint8_t>(temp_level);
    std::cout << "Input coordinates: (" << x << ", " << y << ") level: " << static_cast<int>(level) << std::endl;

    // Encode 2D coordinates into a Morton code
    uint64_t morton_code = morton2d::encode2D(x, y, level);
    std::cout << "Encoded 2D Morton code: " << morton_code << std::endl;

    // Decode the Morton code back into coordinates
    auto decoded_id = morton2d::decode2D(morton_code);
    std::cout << "Decoded 2D coordinates: (" << std::get<0>(decoded_id) << ", " << std::get<1>(decoded_id) << ")"
              << ", level: " << static_cast<int>(std::get<2>(decoded_id)) << std::endl;

    std::cout << " Testing get parent :" << std::endl;
    std::cout << "Enter x coordinate: ";
    std::cin >> x;
    std::cout << "Enter y coordinate: ";
    std::cin >> y;
    std::cout << "Enter refinement level: ";
    std::cin >> temp_level;

    // Explicitly cast the input to uint8_t
    level = static_cast<uint8_t>(temp_level);
    uint64_t morton_code_child = morton2d::encode2D(x, y, level);
    uint64_t morton_parent = morton2d::getParent2D(morton_code_child);
    auto decoded_id_parent = morton2d::decode2D(morton_parent);
    std::cout << "Decoded parent 2D coordinates: (" << std::get<0>(decoded_id_parent) << ", " << std::get<1>(decoded_id_parent) << ")"
              << ", level: " << static_cast<int>(std::get<2>(decoded_id_parent)) << std::endl;

    // Test getChild2D function
    std::cout << "Testing getChild2D function (assuming max depth = 2 (imax = jmax = 4)):" << std::endl;
    std::cout << "Enter x coordinate: ";
    std::cin >> x;
    std::cout << "Enter y coordinate: ";
    std::cin >> y;
    std::cout << "Enter refinement level: ";
    std::cin >> temp_level;

    // Explicitly cast the input to uint8_t
    level = static_cast<uint8_t>(temp_level);
    uint64_t morton_code_parent = morton2d::encode2D(x, y, level);
    auto children = morton2d::getChild2D(morton_code_parent);

    auto child1 = morton2d::decode2D(std::get<0>(children));
    auto child2 = morton2d::decode2D(std::get<1>(children));
    auto child3 = morton2d::decode2D(std::get<2>(children));
    auto child4 = morton2d::decode2D(std::get<3>(children));

    std::cout << "Child 1 coordinates: (" << std::get<0>(child1) << ", " << std::get<1>(child1) << "), level: " << static_cast<int>(std::get<2>(child1)) << std::endl;
    std::cout << "Child 2 coordinates: (" << std::get<0>(child2) << ", " << std::get<1>(child2) << "), level: " << static_cast<int>(std::get<2>(child2)) << std::endl;
    std::cout << "Child 3 coordinates: (" << std::get<0>(child3) << ", " << std::get<1>(child3) << "), level: " << static_cast<int>(std::get<2>(child3)) << std::endl;
    std::cout << "Child 4 coordinates: (" << std::get<0>(child4) << ", " << std::get<1>(child4) << "), level: " << static_cast<int>(std::get<2>(child4)) << std::endl;


    return 0;
}