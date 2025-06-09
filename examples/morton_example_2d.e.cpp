#include <morton/morton.hpp>
#include <iostream>

int main() {
    // Take user input for 2D coordinates
    uint32_t x, y;
    uint8_t level;
    std::cout << "Enter x coordinate: ";
    std::cin >> x;
    std::cout << "Enter y coordinate: ";
    std::cin >> y;
    std::cout << "Enter refinement level: ";
    std::cin >> level;
    std::cout << "Input coordinates: (" << x << ", " << y << ")" <<" level : " << level << std::endl;

    // Encode 2D coordinates into a Morton code
    uint64_t morton_code = morton::encode2D(x, y, level);
    std::cout << "Encoded 2D Morton code: " << morton_code << std::endl;

    // Decode the Morton code back into coordinates
    auto decoded_id = morton::decode2D(morton_code);
    std::cout << "Decoded 2D coordinates: (" << std::get<0>(decoded_id) << ", " << std::get<1>(decoded_id) << ")" << ", level : " << std::get<2>(decoded_id) << std::endl;


    return 0;
}