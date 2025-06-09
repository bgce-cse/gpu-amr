#ifndef AMR_INCLUDED_MORTON
#define AMR_INCLUDED_MORTON
#include <array>
#include <cstdint>
#include <libmorton/morton.h>
#include <vector>
#include <cassert>

constexpr uint32_t MAX_COORD = (1u << 29); // 2^29
constexpr uint8_t MAX_LEVEL = 64;


namespace morton
{

// Encode 2D coordinates into a Morton code
uint64_t encode2D(uint32_t x, uint32_t y, uint8_t level)
{
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

// Get parent code at one level up
uint64_t parent(uint64_t id, int level);

// Get neighbor Morton IDs (only direct neighbors)
std::vector<uint64_t> neighbors2D(uint64_t id, int level);
std::vector<uint64_t> neighbors3D(uint64_t id, int level);

// maybe at some point: encode/decode for nD
// parents and neighbors for nD maybe later,
// but gonna be really tricky
template <int D>
uint64_t encode(const std::array<uint32_t, D>& coords);

template <int D>
std::array<uint32_t, D> decode(uint64_t code);
} // namespace morton

#endif // AMR_INCLUDED_MORTON
