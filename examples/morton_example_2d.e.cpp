#include <morton/morton.hpp>
#include <iostream>
#include <limits> // For debugging purposes

int main() {
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
    uint64_t morton_code = morton::encode2D(x, y, level);
    std::cout << "Encoded 2D Morton code: " << morton_code << std::endl;

    // Decode the Morton code back into coordinates
    auto decoded_id = morton::decode2D(morton_code);
    std::cout << "Decoded 2D coordinates: (" << std::get<0>(decoded_id) << ", " << std::get<1>(decoded_id) << ")"
              << ", level: " << static_cast<int>(std::get<2>(decoded_id)) << std::endl;

    std::cout << " find parent :" << std::endl;
    std::cout << "Enter x coordinate: ";
    std::cin >> x;
    std::cout << "Enter y coordinate: ";
    std::cin >> y;
    std::cout << "Enter refinement level: ";
    std::cin >> temp_level;

    // Explicitly cast the input to uint8_t
    level = static_cast<uint8_t>(temp_level);
    uint64_t morton_code_child = morton::encode2D(x, y, level);
    uint64_t morton_parent = morton::getParent2D(morton_code_child);
    auto decoded_id_parent = morton::decode2D(morton_parent);
    std::cout << "Decoded parent 2D coordinates: (" << std::get<0>(decoded_id_parent) << ", " << std::get<1>(decoded_id_parent) << ")"
              << ", level: " << static_cast<int>(std::get<2>(decoded_id_parent)) << std::endl;



    return 0;
}