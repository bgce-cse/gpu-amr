#include <morton/morton.hpp>
#include <iostream>

int main() {
    // Take user input for 3D coordinates and refinement level
    uint32_t x, y, z;
    int temp_level; // Use an intermediate variable to read the input
    uint8_t level;
    std::cout << "Enter x coordinate: ";
    std::cin >> x;
    std::cout << "Enter y coordinate: ";
    std::cin >> y;
    std::cout << "Enter z coordinate: ";
    std::cin >> z;
    std::cout << "Enter refinement level: ";
    std::cin >> temp_level;

     level = static_cast<uint8_t>(temp_level);
    std::cout << "Input coordinates: (" << x << ", " << y << ", " << z << "), level: " << static_cast<int>(level) << std::endl;

    // Encode 3D coordinates into a Morton code
    uint64_t morton_code = morton::encode3D(x, y, z, level);
    std::cout << "Encoded 3D Morton code: " << morton_code << std::endl;

    // Decode the Morton code back into coordinates
    auto decoded_coords = morton::decode3D(morton_code);
    std::cout << "Decoded 3D coordinates: (" << std::get<0>(decoded_coords) << ", "
              << std::get<1>(decoded_coords) << ", " << std::get<2>(decoded_coords) << "), level: "
              << static_cast<int>(std::get<3>(decoded_coords)) << std::endl;

    return 0;
}