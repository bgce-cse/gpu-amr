#ifndef AMR_INCLUDED_MORTON
#define AMR_INCLUDED_MORTON

#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <libmorton/morton.h>
#include <utility>

namespace amr::ndt::morton
{

// Base template - not implemented to force specialization
template <std::unsigned_integral auto Depth, std::unsigned_integral auto Dimension>
class morton_id;

// 2D Specialization
template <std::unsigned_integral auto Depth>
class morton_id<Depth, 2u>
{
public:
    enum direction
    {
        left,
        right,
        top,
        bottom
    };

    using depth_t     = decltype(Depth);
    using id_t        = uint64_t;
    using coord_array = std::array<uint32_t, 2>;
    using offset_t = uint32_t;

    static constexpr auto s_depth = Depth;
    static constexpr auto s_dim   = 2u;

    static constexpr offset_t offset_i(id_t id){
        id_t morton_id = id >> 6;
        return morton_id & 0x3;
    }

    static constexpr id_t zeroth_generation(){
        return 0;
    } 
    static constexpr bool less(id_t lhs, id_t rhs) noexcept
    {
        return lhs < rhs; 
    }
    static constexpr bool equal(id_t lhs, id_t rhs) noexcept
    {
        return lhs == rhs;
    }
    static constexpr bool isvalid_coord(coord_array coordinates, uint8_t level)
    {
        uint32_t grid_size = 1u << (s_depth - level);
        uint32_t max_coord = 1u << s_depth;
        return (coordinates[0] % grid_size == 0) && (coordinates[1] % grid_size == 0) &&
               (coordinates[0] < max_coord) && (coordinates[1] < max_coord);
    }

    static constexpr id_t parent_of(id_t morton_code)
    {
        auto [coords, level] = decode(morton_code);
        assert(level > 0 && "Root has no parent");
        uint32_t offset = 1u << (s_depth - level);
        coords[0] &= ~offset;
        coords[0] &= ~offset;
        assert(isvalid_coord(coords, level) && "invalid parent coordiantes");

        return encode(coords, level - 1);
    }

    static constexpr std::pair<coord_array, uint8_t> decode(id_t id)
    {
        uint8_t  level     = id & 0x3F;
        uint64_t morton_id = id >> 6;

        uint_fast32_t x, y;
        libmorton::morton2D_64_decode(morton_id, x, y);

        return {
            { static_cast<uint32_t>(x), static_cast<uint32_t>(y) },
            level
        };
    }

    static constexpr id_t encode(coord_array const& coords, uint8_t level)
    {
        uint64_t morton_id = libmorton::morton2D_64_encode(coords[0], coords[1]);
        return (morton_id << 6) | (level & 0x3F);
    }

    static constexpr id_t child_of(id_t parent_id)
    {
        auto [coords, level] = decode(parent_id); 
        assert(level < s_depth && "Cell already at max level");  

        return parent_id + 1;  
    }

    static constexpr id_t getNeighbor(id_t morton, direction dir) 
    {
        auto [coords, level] = decode(morton);  
        uint32_t x_coord = coords[0];           
        uint32_t y_coord = coords[1];           
        uint32_t offset = 1u << (s_depth - level);  

        switch (dir)
        {
            case direction::left:   x_coord -= offset; break;
            case direction::right:  x_coord += offset; break;
            case direction::bottom: y_coord += offset; break;
            case direction::top:    y_coord -= offset; break;
            default: assert(false); break;
        }

        if (!isvalid_coord({x_coord, y_coord}, level))  
        {
            return 0;  
        }

        return encode({x_coord, y_coord}, level);  
    }
};

// 3D Specialization
// template <std::unsigned_integral auto Depth>
// class morton_id<Depth, 3>
// {
// public:
//     using depth_t     = decltype(Depth);
//     using id_t        = uint64_t;
//     using coord_array = std::array<uint32_t, 3>;

//     static constexpr auto s_depth = Depth;
//     static constexpr auto s_dim   = 3u;

//     static constexpr id_t parent_of(id_t morton_code)
//     {
//         auto [coords, level] = decode(morton_code);
//         assert(level > 0 && "Root has no parent");

//         // Shift coordinates to parent level
//         coords[0] >>= 1;
//         coords[1] >>= 1;
//         coords[2] >>= 1;

//         return encode(coords, level - 1);
//     }

//     static constexpr std::pair<coord_array, uint8_t> decode(id_t id)
//     {
//         uint8_t  level     = id & 0x3F;
//         uint64_t morton_id = id >> 6;

//         uint_fast32_t x, y, z;
//         libmorton::morton3D_64_decode(morton_id, x, y, z);

//         return {
//             { static_cast<uint32_t>(x),
//              static_cast<uint32_t>(y),
//              static_cast<uint32_t>(z) },
//             level
//         };
//     }

//     static constexpr id_t encode(coord_array const& coords, uint8_t level)
//     {
//         uint64_t morton_id =
//             libmorton::morton3D_64_encode(coords[0], coords[1], coords[2]);
//         return (morton_id << 6) | (level & 0x3F);
//     }
// };

} // namespace amr::ndt::morton

#endif // AMR_INCLUDED_MORTON