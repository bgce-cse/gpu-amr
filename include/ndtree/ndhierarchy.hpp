#ifndef AMR_INCLUDED_NDHIERARCHY
#define AMR_INCLUDED_NDHIERARCHY

#include "allocators/allocator_concepts.hpp"
#include "allocators/free_list_buffer_allocator.hpp"
#include "ndtree/ndutils.hpp"
#include "utility/error_handling.hpp"
#include <algorithm>
#include <bitset>
#include <concepts>
#include <cstdint>
#include <type_traits>
#include <vector>

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
    static_assert(s_hierarchy_bits <= sizeof(mask_t) * 8);
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

private:
    constexpr hierarchy_id(depth_t generation, id_t id, mask_t fanout_id) noexcept
        : m_generation{ generation }
        , m_id{ id.to_ullong() +
                (fanout_id <<= ((s_depth - m_generation) * s_generation_bits)) }
    {
        assert(m_generation < s_depth);
    }

public:
    constexpr hierarchy_id(hierarchy_id parent_id, mask_t fanout_id) noexcept
        : m_generation{ parent_id.generation() + 1 }
        , m_id{ parent_id.id().to_ullong() +
                (fanout_id <<= ((s_depth - m_generation) * s_generation_bits)) }
    {
        assert(m_generation < s_depth);
    }

    static constexpr auto zeroth_generation() noexcept -> hierarchy_id
    {
        return hierarchy_id({}, {}, {});
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

template <
    typename T,
    std::unsigned_integral auto Dim,
    std::unsigned_integral auto Fanout,
    std::unsigned_integral auto Max_Depth,
    std::unsigned_integral      Index_Type = std::uint32_t>
class ndhierarchy
{
    static_assert(Dim > 1);
    static_assert(Fanout > 1);

public:
    using index_t       = Index_Type;
    using size_type     = index_t;
    using depth_t       = decltype(Max_Depth);
    using value_type    = T;
    using pointer       = value_type*;
    using const_pointer = value_type const*;

    static constexpr auto s_dim       = Dim;
    static constexpr auto s_max_depth = Max_Depth;
    static constexpr auto s_1d_fanout = Fanout;
    static constexpr auto s_nd_fanout = utility::cx_functions::pow(s_1d_fanout, s_dim);
    using hierarchy_id_t              = hierarchy_id<s_nd_fanout, s_max_depth>;
    using mask_t                      = typename hierarchy_id_t::mask_t;

    struct block_pointer
    {
        hierarchy_id_t id;
        pointer        ptr;

        auto operator<(block_pointer const& other) const -> bool
        {
            if (id.id().to_ullong() < other.id.id().to_ullong())
                return true;
            else if (id.id().to_ullong() > other.id.id().to_ullong())
                return false;
            else
                return id.generation() < other.id.generation();
        }
    };

    static constexpr auto s_block_size = sizeof(block_pointer);
    using allocator_t =
        allocator::free_list_buffer_allocator<s_block_size, alignof(block_pointer)>;
    using container_t = std::vector<block_pointer>;

public:
    ndhierarchy() noexcept
        : m_allocator()
        , m_blocks()
    {
        auto  p    = (pointer)m_allocator.allocate_one();
        auto* root = new (p) block_pointer(hierarchy_id_t::zeroth_generation(), p);
        m_blocks.push_back(*root);
    }

    inline auto fragment(hierarchy_id_t const& parent_id, const mask_t fanout_id)
        -> block_pointer
    {
        auto  p     = (pointer)m_allocator.allocate_one();
        auto* block = new (p) block_pointer(hierarchy_id_t{ parent_id, fanout_id }, p);
        m_blocks.push_back(*block);
        return *block;
    }

    inline auto remove(hierarchy_id_t const& h) noexcept
    {
        auto parent =
            std::ranges::remove_if(m_blocks, [h](auto const& e) { return e.id == h; });
    }

    [[nodiscard]]
    inline auto members() const noexcept -> auto const&
    {
        return m_blocks;
    }

private:
    allocator_t m_allocator;
    container_t m_blocks;
};

} // namespace amr::ndt::hierarchy

#endif // AMR_INCLUDED_NDT_HIERARCHY
