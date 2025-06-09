#include <morton/morton.hpp>
#include <iostream>

int main() {
    // Take user input for 3D coordinates
    uint32_t x, y, z;
    std::cout << "Enter x coordinate: ";
    std::cin >> x;
    std::cout << "Enter y coordinate: ";
    std::cin >> y;
    std::cout << "Enter z coordinate: ";
    std::cin >> z;

    std::cout << "Input coordinates: (" << x << ", " << y << ", " << z << ")" << std::endl;

    // Encode 3D coordinates into a Morton code
    uint64_t morton_code = morton::encode3D(x, y, z);
    std::cout << "Encoded 3D Morton code: " << morton_code << std::endl;

    // Decode the Morton code back into coordinates
    auto decoded_coords = morton::decode3D(morton_code);
    std::cout << "Decoded 3D coordinates: (" << std::get<0>(decoded_coords) << ", "
              << std::get<1>(decoded_coords) << ", " << std::get<2>(decoded_coords) << ")" << std::endl;

    return 0;
}