#ifndef AMR_INCLUDED_NDHIERARCHY
#define AMR_INCLUDED_NDHIERARCHY

#include "utility/constexpr_functions.hpp"
#include <bitset>
#include <climits>
#include <concepts>
#include <iostream>

namespace amr::ndt::hierarchy
{

template <
    std::unsigned_integral auto Dim,
    std::unsigned_integral auto Fanout,
    std::unsigned_integral auto Depth>
class hierarchical_prefix_id
{
    static_assert(Dim > 1);
    static_assert(Fanout > 1);

public:
    using size_type   = std::uint32_t;
    using mask_t      = unsigned long long;
    using offset_t    = mask_t;
    using direction_t = int; // TODO: Make an enum

private:
    static constexpr size_type s_dim       = Dim;
    static constexpr size_type s_1d_fanout = Fanout;
    static constexpr size_type s_nd_fanout =
        utility::cx_functions::pow(s_1d_fanout, s_dim);
    static constexpr auto s_max_depth = Depth;

    static constexpr auto s_generation_bits =
        utility::cx_functions::bits_for(s_nd_fanout);
    static constexpr auto s_hierarchy_bits = s_generation_bits * s_max_depth;
    static_assert(s_hierarchy_bits <= sizeof(mask_t) * CHAR_BIT);
    using id_t = std::bitset<s_hierarchy_bits>;

public:
    static constexpr std::array<mask_t, s_max_depth + 1> s_generation_masks =
        []() constexpr
    {
        auto                                mask = ~(~0ull << s_generation_bits);
        std::array<mask_t, s_max_depth + 1> masks{};
        for (auto i = s_max_depth; i != 0; --i)
        {
            masks[i] = mask;
            mask <<= s_generation_bits;
        }
        return masks;
    }();

    static constexpr std::array<mask_t, s_max_depth> s_predecessor_masks = []() constexpr
    {
        std::array<mask_t, Depth> masks{};
        for (auto i = 1; i != Depth; ++i)
        {
            masks[i] = masks[i - 1] ^ s_generation_masks[i];
        }
        return masks;
    }();

private:
    constexpr hierarchical_prefix_id(size_type generation, id_t id) noexcept
        : m_generation{ generation }
        , m_id{ id }
    {
        assert(m_generation < s_max_depth);
    }

    constexpr hierarchical_prefix_id(
        size_type generation,
        id_t      id,
        mask_t    fanout_id
    ) noexcept
        : m_generation{ generation }
        , m_id{ id.to_ullong() |
                (fanout_id << ((s_max_depth - m_generation) * s_generation_bits)) }
    {
        assert(m_generation < s_max_depth);
    }

public:
    [[nodiscard]]
    static constexpr auto dimension() noexcept -> size_type
    {
        return s_dim;
    }

    [[nodiscard]]
    static constexpr auto fanout() noexcept -> size_type
    {
        return s_1d_fanout;
    }

    [[nodiscard]]
    static constexpr auto nd_fanout() noexcept -> size_type
    {
        return s_nd_fanout;
    }

    [[nodiscard]]
    static constexpr auto max_depth() noexcept -> size_type
    {
        return s_max_depth;
    }

    [[nodiscard]]
    static constexpr auto zeroth_generation() noexcept -> hierarchical_prefix_id
    {
        return hierarchical_prefix_id({}, {}, {});
    }

    [[nodiscard]]
    static constexpr auto parent_of(hierarchical_prefix_id const& id) noexcept
        -> hierarchical_prefix_id
    {
        assert(id.m_generation > 0);
        const auto p = hierarchical_prefix_id(
            id.m_generation - 1,
            id.m_id.to_ullong() & s_predecessor_masks[id.m_generation - 1]
        );
        std::cout << "Parent of " << id.m_id.to_string() << " is " << p.m_id.to_string()
                  << '\n';
        return p;
    }

    [[nodiscard]]
    static constexpr auto child_of(const hierarchical_prefix_id id) noexcept
        -> hierarchical_prefix_id
    {
        assert(id.m_generation < s_max_depth);
        const auto p = hierarchical_prefix_id{ id.m_generation + 1, id.m_id.to_ullong() };
        std::cout << "Gen: " << id.m_generation << ", Child 0 of " << id.m_id.to_string()
                  << " is " << p.m_id.to_string() << '\n';
        return p;
    }

    [[nodiscard]]
    static constexpr auto
        neighbour_at(hierarchical_prefix_id const& id, direction_t d) noexcept
        -> hierarchical_prefix_id
    {
        std::cout << "Not implemented yet, i dont know how to find neighbour" << d
                  << '\n';
        return id;
    }

    [[nodiscard]]
    static constexpr auto offset_of(hierarchical_prefix_id const& id) noexcept -> offset_t
    {
        return (
            (id.m_id.to_ullong() & s_generation_masks[id.m_generation]) >>
            ((s_max_depth - id.m_generation) * s_generation_bits)
        );
    }

    [[nodiscard]]
    static constexpr auto
        offset(const hierarchical_prefix_id first_sibling, const offset_t offset) noexcept
        -> hierarchical_prefix_id
    {
        const auto sibling = hierarchical_prefix_id(
            first_sibling.m_generation, first_sibling.m_id, offset
        );
        std::cout << "Sibling " << offset << " of " << first_sibling.m_id.to_string()
                  << " is " << sibling.m_id.to_string() << '\n';
        return sibling;
    }

    [[nodiscard]]
    constexpr auto id() const noexcept -> id_t const&
    {
        return m_id;
    }

    [[nodiscard]]
    constexpr auto generation() const noexcept -> size_type
    {
        return m_generation;
    }

    [[nodiscard]]
    auto operator==(hierarchical_prefix_id const&) const noexcept -> bool = default;

private:
    size_type m_generation;
    id_t      m_id;
};

template <
    std::unsigned_integral auto Dim,
    std::unsigned_integral auto Fanout,
    std::unsigned_integral auto Depth>
auto operator<(
    hierarchical_prefix_id<Dim, Fanout, Depth> const& idx_a,
    hierarchical_prefix_id<Dim, Fanout, Depth> const& idx_b
) noexcept
{
    if (idx_a.id().to_ullong() < idx_b.id().to_ullong())
        return true;
    else if (idx_a.id().to_ullong() > idx_b.id().to_ullong())
        return false;
    else
        return idx_a.generation() < idx_b.generation();
}

} // namespace amr::ndt::hierarchy

#endif // AMR_INCLUDED_NDT_HIERARCHY
