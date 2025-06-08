#ifndef MORTON
#define MORTON

#include <cstdint>
#include <array>
#include <vector>

namespace morton {

    // Encode 2D coordinates into a Morton code
    uint64_t encode2D(uint32_t x, uint32_t y);

    // Decode a 2D Morton code into coordinates
    std::pair<uint32_t, uint32_t> decode2D(uint64_t id);

    // Encode 3D coordinates into a Morton code
    uint64_t encode3D(uint32_t x, uint32_t y, uint32_t z);

    // Decode a 3D Morton code into coordinates
    std::tuple<uint32_t, uint32_t, uint32_t> decode3D(uint64_t id);

    // Get parent code at one level up
    uint64_t parent(uint64_t id, int level);

    // Get neighbor Morton IDs (only direct neighbors)
    std::vector<uint64_t> neighbors2D(uint64_t id, int level);
    std::vector<uint64_t> neighbors3D(uint64_t id, int level);

    // maybe at some point: encode/decode for nD 
    // parents and neighbors for nD maybe later, 
    // but gonna be really tricky
    template<int D>
    uint64_t encode(const std::array<uint32_t, D>& coords);

    template<int D>
    std::array<uint32_t, D> decode(uint64_t code);
}

#endif // MORTON_H
