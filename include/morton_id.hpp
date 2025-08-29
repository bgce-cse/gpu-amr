#pragma once

#include "tree_types.hpp"
#include "utility/error_handling.hpp"
#include <array>
#include <bitset>
#include <cassert>
#include <climits>
#include <cstdint>
#include <iostream>
#include <morton.h>
#include <optional>
#include <utility>

namespace morton
{

// Base template - not implemented to force specialization
template <
    std::unsigned_integral auto Depth, std::unsigned_integral auto Dimension>
class morton_id;

// 2D Specialization
template <std::unsigned_integral auto Depth>
class morton_id<Depth, 2u>
{
  public:
    enum direction
    {
        N,
        S,
        E,
        W,
        NE,
        NW,
        SE,
        SW,
    };

    using depth_t = decltype(Depth);
    using mask_t = uint64_t;
    using coord_array = std::array<uint32_t, 2>;
    using rel_coord_array = std::array<float, 2>;
    using offset_t = uint32_t;
    using size_type = uint32_t;
    using direction_t = direction;
    using level_t = uint8_t;

    static constexpr size_type s_max_depth = Depth;
    inline static size_type s_depth = s_max_depth;
    static constexpr size_type s_dim = 2u;

    constexpr morton_id()
        : m_id{}
    {
    }
    constexpr morton_id(mask_t i)
        : m_id{i}
    {
    }

    static constexpr offset_t offset_of(morton_id i)
    {
        auto [coords, level] = decode(i);
        uint32_t delta_x_y = 1u << (s_depth - level + 1);
        offset_t x_offset = coords[0] % delta_x_y == 0 ? 0 : 1;
        offset_t y_offset = coords[1] % delta_x_y == 0 ? 0 : 2;
        return x_offset + y_offset;
    }

    static constexpr morton_id offset(morton_id i, offset_t offset)
    {
        // Assume offset in {0, 1, 2, 3}
        // Assume id corresponds to zero sibling (bits 6-7 are 0)
        assert(
            offset_of(i) == 0 &&
            "This function assumes it is called for zero sibling"
        );
        assert(offset <= 3 && "Offset must be 0, 1, 2, or 3");
        auto [coords, level] = decode(i);
        uint32_t delta_x_y = 1u << (s_depth - level);
        offset_t x_offset = offset % 2;
        offset_t y_offset = offset > 1 ? 1 : 0;

        coords[0] += x_offset * delta_x_y;
        coords[1] += y_offset * delta_x_y;

        auto sibling_id = encode(coords, level);

        return morton_id(sibling_id);
    }

    static constexpr morton_id root()
    {
        return morton_id(0);
    }

    static constexpr size_t bits()
    {
        return sizeof(mask_t) * CHAR_BIT;
    }

    static constexpr bool isvalid_coord(coord_array coordinates, level_t level)
    {
        uint32_t grid_size = 1u << (s_depth - level);
        uint32_t max_coord = 1u << s_depth;
        return (coordinates[0] % grid_size == 0) &&
               (coordinates[1] % grid_size == 0) &&
               (coordinates[0] < max_coord) && (coordinates[1] < max_coord);
    }

    static constexpr morton_id parent_of(morton_id morton_code)
    {
        auto [coords, level] = decode(morton_code);
        assert(level > 0 && "Root has no parent");
        uint32_t offset = 1u << (s_depth - level);
        coords[0] &= ~offset;
        coords[1] &= ~offset;
        assert(
            isvalid_coord(coords, level - 1) && "invalid parent coordiantes"
        );

        return morton_id(encode(coords, level - 1));
    }

    static constexpr std::pair<coord_array, level_t> decode(morton_id const& i)
    {
        level_t level = i.m_id & 0x3F;
        mask_t morton_id = i.m_id >> 6;

        uint_fast32_t x, y;
        libmorton::morton2D_64_decode(morton_id, x, y);

        return {{static_cast<uint32_t>(x), static_cast<uint32_t>(y)}, level};
    }

    static constexpr mask_t encode(coord_array const& coords, level_t level)
    {
        assert(
            isvalid_coord(coords, level) &&
            "invalid coordiantes and level combination"
        );

        uint64_t morton_id =
            libmorton::morton2D_64_encode(coords[0], coords[1]);
        return (morton_id << 6) | (level & 0x3F);
    }

    static constexpr rel_coord_array rel_coords(morton_id id)
    {
        auto [coords, _] = decode(id);
        const auto max_value = (float)(1u << max_depth());
        return rel_coord_array{
            (float)coords[0] / max_value, (float)coords[1] / max_value
        };
    }

    static constexpr morton_id child_of(
        morton_id const& parent_id, offset_t off
    )
    {
        [[maybe_unused]] const auto lvl = level(parent_id);
        assert(lvl < s_depth && "Cell already at max level");
        auto sibling = offset(parent_id.m_id + 1, off);

        return sibling;
    }

    static constexpr auto level(morton_id i)
    {
        auto [_, level] = decode(i);
        return level;
    }

    static constexpr std::optional<morton_id> neighbour_at(
        morton_id morton, direction dir
    )
    {
        auto [coords, level] = decode(morton);
        uint32_t x_coord = coords[0];
        uint32_t y_coord = coords[1];
        uint32_t offset = 1u << (s_depth - level);

        switch (dir)
        {
        case direction::W:
        case direction::NW:
        case direction::SW:
            x_coord -= offset;
            break;

        case direction::E:
        case direction::NE:
        case direction::SE:
            x_coord += offset;
            break;

        case direction_t::N:
        case direction_t::S:
        default:
            break;
        }

        switch (dir)
        {
        case direction::S:
        case direction::SE:
        case direction::SW:
            y_coord += offset;
            break;

        case direction::N:
        case direction::NE:
        case direction::NW:
            y_coord -= offset;
            break;

        case direction_t::E:
        case direction_t::W:
        default:
            break;
        }

        if (!isvalid_coord({x_coord, y_coord}, level))
        {
            return std::nullopt;
        }

        return morton_id(encode({x_coord, y_coord}, level));
    }

    constexpr auto id() const noexcept
    {
        return m_id;
    }

    constexpr auto repr() const noexcept -> std::string
    {
        return std::bitset<bits()>(m_id).to_string();
    }

    constexpr auto level() const noexcept
    {
        auto [_, level] = decode(*this);
        return level;
    }

    static constexpr auto dimension()
    {
        return s_dim;
    }

    static constexpr size_type nd_fanout()
    {
        return 4;
    }

    static constexpr size_type fanout()
    {
        return 2;
    }

    inline static void set_max_depth(size_type depth)
    {
        if (depth > 0)
        {
            s_depth = depth;
        }
        else
        {
            s_depth = s_max_depth;
        }
    }
    inline static size_type max_depth()
    {
        return s_depth;
    }

    auto operator==(morton_id const&) const noexcept -> bool = default;

  private:
    mask_t m_id;
};

template <std::unsigned_integral auto Depth>
auto operator<(
    morton_id<Depth, 2u> const& idx_a, morton_id<Depth, 2u> const& idx_b
) noexcept
{
    return idx_a.id() < idx_b.id();
}

} // namespace morton

namespace std
{

template <
    std::unsigned_integral auto Depth, std::unsigned_integral auto Dimension>
struct hash<morton::morton_id<Depth, Dimension>>
{
    [[nodiscard]]
    auto operator()(morton::morton_id<Depth, Dimension> const& id
    ) const noexcept -> std::size_t
    {
        return id.id();
    }
};

} // namespace std
