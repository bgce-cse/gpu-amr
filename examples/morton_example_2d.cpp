#include <morton/morton.hpp>
#include <iostream>

int main() {
    // Take user input for 2D coordinates
    uint32_t x, y;
    std::cout << "Enter x coordinate: ";
    std::cin >> x;
    std::cout << "Enter y coordinate: ";
    std::cin >> y;

    std::cout << "Input coordinates: (" << x << ", " << y << ")" << std::endl;

    // Encode 2D coordinates into a Morton code
    uint64_t morton_code = morton::encode2D(x, y);
    std::cout << "Encoded 2D Morton code: " << morton_code << std::endl;

    // Decode the Morton code back into coordinates
    auto decoded_coords = morton::decode2D(morton_code);
    std::cout << "Decoded 2D coordinates: (" << decoded_coords.first << ", " << decoded_coords.second << ")" << std::endl;

    return 0;
}