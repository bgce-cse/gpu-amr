#ifndef AMR_INCLUDED_MORTON
#define AMR_INCLUDED_MORTON
#include <array>
#include <cstdint>
#include <libmorton/morton.h>
#include <vector>
#include <cassert>

constexpr uint32_t MAX_COORD = (1u << 29); // 2^29
constexpr uint8_t MAX_LEVEL = 64;
constexpr uint8_t MAX_DEPTH = 2;


namespace morton3d
{

// Encode 3D coordinates into a Morton code with refinement level
uint64_t encode3D(uint32_t x, uint32_t y, uint32_t z, uint8_t level)
{
    assert(x < (1u << 19) && "coordinate values exceed 19-bit limit");
    assert(y < (1u << 19) && "coordinate values exceed 19-bit limit");
    assert(z < (1u << 19) && "coordinate values exceed 19-bit limit");
    assert(level < MAX_LEVEL && "refinement level exceeds 6-bit limit");
    uint64_t morton_id = libmorton::morton3D_64_encode(x, y, z);
    return (morton_id << 6) | (level & 0x3F);
}

// Decode a 3D Morton code into coordinates and refinement level
std::tuple<uint32_t, uint32_t, uint32_t, uint8_t> decode3D(uint64_t id)
{
    uint8_t level = id & 0x3F; 
    uint64_t morton_id = id >> 6;
    uint_fast32_t x, y, z;
    libmorton::morton3D_64_decode(morton_id, x, y, z); // Decode into uint_fast32_t
    return std::make_tuple(
        static_cast<uint32_t>(x),
        static_cast<uint32_t>(y),
        static_cast<uint32_t>(z),
        level
    );
}


uint64_t getParent3D(uint64_t morton) {
    
    auto [x_fine, y_fine,z_fine, level] = morton3d::decode3D(morton);
    assert(level > 0); // root has no parent
    // Shift one level coarser

    x_fine &= ~1;
    y_fine &= ~1;
    z_fine &= ~1;

    uint64_t parent_morton = morton3d::encode3D(x_fine, y_fine,z_fine, level -1);
    return parent_morton;
}


} // namespace morton

#endif // AMR_INCLUDED_MORTON
