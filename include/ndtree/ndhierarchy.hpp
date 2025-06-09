#ifndef AMR_INCLUDED_NDHIERARCHY
#define AMR_INCLUDED_NDHIERARCHY

#include "utility/constexpr_functions.hpp"
#include <bitset>
#include <climits>
#include <concepts>
#include <iostream>

namespace amr::ndt::hierarchy
{

template <std::unsigned_integral auto Fanout, std::unsigned_integral auto Depth>
class hierarchy_id
{
public:
    using depth_t                           = decltype(Depth);
    using mask_t                            = unsigned long long;
    static constexpr auto s_fanout          = Fanout;
    static constexpr auto s_depth           = Depth;
    static constexpr auto s_generation_bits = utility::cx_functions::bits_for(s_fanout);
    static constexpr auto s_hierarchy_bits  = s_generation_bits * s_depth;
    static_assert(s_hierarchy_bits <= sizeof(mask_t) * CHAR_BIT);
    using id_t = std::bitset<s_hierarchy_bits>;

    static constexpr std::array<mask_t, Depth + 1> s_generation_masks = []() constexpr
    {
        auto                          mask = ~(~0ull << s_generation_bits);
        std::array<mask_t, Depth + 1> masks{};
        for (auto i = s_depth; i != 0; --i)
        {
            masks[i] = mask;
            mask <<= s_generation_bits;
        }
        return masks;
    }();

    static constexpr std::array<mask_t, Depth> s_predecessor_masks = []() constexpr
    {
        std::array<mask_t, Depth> masks{};
        for (auto i = 1; i != Depth; ++i)
        {
            masks[i] = masks[i - 1] ^ s_generation_masks[i];
        }
        return masks;
    }();

private:
    constexpr hierarchy_id(depth_t generation, id_t id) noexcept
        : m_generation{ generation }
        , m_id{ id }
    {
        assert(m_generation < s_depth);
    }

    constexpr hierarchy_id(depth_t generation, id_t id, mask_t fanout_id) noexcept
        : m_generation{ generation }
        , m_id{ id.to_ullong() |
                (fanout_id <<= ((s_depth - m_generation) * s_generation_bits)) }
    {
        assert(m_generation < s_depth);
    }

public:
    [[nodiscard]]
    static constexpr auto zeroth_generation() noexcept -> hierarchy_id
    {
        return hierarchy_id({}, {}, {});
    }

    [[nodiscard]]
    static constexpr auto parent_of(hierarchy_id const& id) noexcept -> hierarchy_id
    {
        assert(id.m_generation > 0);
        const auto p = hierarchy_id(
            id.m_generation - 1, id.m_id & s_predecessor_masks[id.m_generation - 1]
        );
        std::cout << "Parent of " << id.m_id.to_string() << " is " << p.m_id.to_string()
                  << '\n';
        return p;
    }

    [[nodiscard]]
    static constexpr auto child_of(const hierarchy_id id, const mask_t fanout_id) noexcept
        -> hierarchy_id
    {
        assert(id.m_generation < s_depth);
        const auto p = hierarchy_id{
            id.m_generation + 1,
            id.m_id.to_ullong() ^
                (fanout_id << ((s_depth - (id.m_generation + 1)) * s_generation_bits))
        };
        std::cout << id.m_generation << ", Child " << fanout_id << " of "
                  << id.m_id.to_string() << " is " << p.m_id.to_string() << '\n';
        return p;
    }

    [[nodiscard]]
    constexpr auto id() const noexcept -> id_t const&
    {
        return m_id;
    }

    [[nodiscard]]
    constexpr auto generation() const noexcept -> depth_t
    {
        return m_generation;
    }

    [[nodiscard]]
    constexpr auto generation_id() const noexcept -> mask_t
    {
        return (
            (m_id.to_ullong() & s_generation_masks[m_generation]) >>
            ((s_depth - m_generation) * s_generation_bits)
        );
    }

private:
    depth_t m_generation;
    id_t    m_id;
};

} // namespace amr::ndt::hierarchy

#endif // AMR_INCLUDED_NDT_HIERARCHY
