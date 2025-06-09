#ifndef AMR_INCLUDED_MORTON
#define AMR_INCLUDED_MORTON
#include <array>
#include <cstdint>
#include <libmorton/morton.h>
#include <vector>
#include <cassert>
#include <iostream>
#include <optional>

constexpr uint32_t MAX_COORD = (1u << 29); // 2^29
constexpr uint8_t MAX_LEVEL = 64;
constexpr uint8_t MAX_DEPTH = 2;


enum  direction
{
    left,
    right,
    top,
    bottom
};


namespace morton2d
{
bool isvalid_coord(uint32_t x, uint32_t y, uint8_t level) {
    uint32_t grid_size = 1u << (MAX_DEPTH - level);
    uint32_t max_coord = 1u << MAX_DEPTH;
    return (x % grid_size == 0) && (y % grid_size == 0) && (x < max_coord ) && (y < max_coord );
}

// Encode 2D coordinates into a Morton code
uint64_t encode2D(uint32_t x, uint32_t y, uint8_t level)
{
    assert(isvalid_coord(x,y,level) && "invalid combination of coordinates and level. Coordinates need to be multiple of 2^(max depth (harcoded to 2 rn for testing) - level )");
    assert(x < MAX_COORD && "coordinate values exceed 29-bit limit");
    assert(y < MAX_COORD && "coordinate values exceed 29-bit limit");
    assert(level < MAX_LEVEL && "refinement level exceeds 6-bit limit");
    uint64_t morton_id = libmorton::morton2D_64_encode(x, y);
    return (morton_id << 6) | (level & 0x3F);
}

// Decode a 2D Morton code into coordinates
std::tuple<uint32_t, uint32_t, uint8_t> decode2D(uint64_t id)
{
    uint8_t level = id & 0x3F; 
    uint64_t morton_id = id >> 6;
    uint_fast32_t x, y;
    libmorton::morton2D_64_decode(morton_id, x, y); // Decode into uint_fast32_t
    return std::make_tuple(static_cast<uint32_t>(x), static_cast<uint32_t>(y), level);
}


uint64_t getParent2D(uint64_t morton) {
    
    auto [x_fine, y_fine, level] = morton2d::decode2D(morton);
    assert(level > 0); // root has no parent
    // Shift one level coarser
    uint32_t offset = 1u << (MAX_DEPTH - level);
    x_fine &= ~offset;
    y_fine &= ~offset;
    // std::cout << "x_parent " << x_fine << "y parent" << y_fine << std::endl; 

    uint64_t parent_morton = morton2d::encode2D(x_fine, y_fine, level -1);
    return parent_morton;
}

bool isSibling(uint64_t morton1, uint64_t morton2 ){
    return morton2d::getParent2D(morton1) == morton2d::getParent2D(morton2); 
}


std::tuple<uint64_t, uint64_t, uint64_t, uint64_t> getChild2D(uint64_t morton) {
    auto decoded_id = morton2d::decode2D(morton);
    uint32_t x_coord = std::get<0>(decoded_id);
    uint32_t y_coord = std::get<1>(decoded_id);
    uint8_t level = std::get<2>(decoded_id);

    assert(level < MAX_DEPTH && "Cell already at max level - no more child possible");

    // Calculate the offset for child cells at the next refinement level
    uint32_t offset = 1u << (MAX_DEPTH - level - 1);

    // Encode child Morton codes
    uint64_t morton1 = morton2d::encode2D(x_coord, y_coord, level + 1);
    uint64_t morton2 = morton2d::encode2D(x_coord + offset, y_coord, level + 1);
    uint64_t morton3 = morton2d::encode2D(x_coord, y_coord + offset, level + 1);
    uint64_t morton4 = morton2d::encode2D(x_coord + offset, y_coord + offset, level + 1);

    return {morton1, morton2, morton3, morton4};
}

std::optional<uint64_t> getNeighbor(uint64_t morton, direction dir) {
    auto decoded_id = morton2d::decode2D(morton);
    uint32_t x_coord = std::get<0>(decoded_id);
    uint32_t y_coord = std::get<1>(decoded_id);
    uint8_t level = std::get<2>(decoded_id);
    uint32_t offset = 1u << (MAX_DEPTH - level);

    switch (dir) {
        case direction::left: x_coord -= offset; break;
        case direction::right: x_coord += offset; break;
        case direction::top: y_coord += offset; break;
        case direction::bottom: y_coord -= offset; break;
        default: assert(false); break;
    }

    // std::cout << "neighbor coordinates on same level :" << x_coord<< ", " << y_coord << std::endl;
    if (!morton2d::isvalid_coord(x_coord, y_coord, level)) {
        return std::nullopt; // No neighbor exists
    }

    return morton2d::encode2D(x_coord, y_coord, level);
}


} // namespace morton

#endif // AMR_INCLUDED_MORTON
