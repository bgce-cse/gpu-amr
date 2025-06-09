#ifndef AMR_INCLUDED_NDTREE
#define AMR_INCLUDED_NDTREE

#include "allocators/free_list_buffer_allocator.hpp"
#include "ndhierarchy.hpp"
#include "utility/constexpr_functions.hpp"
#include <concepts>
#include <cstdint>

namespace amr::ndt::tree
{

template <
    typename T,
    std::unsigned_integral auto Dim,
    std::unsigned_integral auto Fanout,
    std::unsigned_integral auto Max_Depth,
    std::unsigned_integral      Index_Type = std::uint32_t>
class ndtree
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
    using hierarchy_id_t              = hierarchy::hierarchy_id<s_nd_fanout, s_max_depth>;
    using mask_t                      = typename hierarchy_id_t::mask_t;

    static constexpr auto s_block_size = sizeof(value_type) * s_nd_fanout;
    using allocator_t =
        allocator::free_list_buffer_allocator<s_block_size, alignof(value_type)>;

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

    using container_t = std::vector<block_pointer>;

public:
    ndtree() noexcept
        : m_allocator()
        , m_blocks()
    {
        const auto p = (pointer)m_allocator.allocate_one();
        m_blocks.emplace_back(hierarchy_id_t::zeroth_generation(), p);
    }

    inline auto fragment(hierarchy_id_t const& parent_id, const mask_t fanout_id)
        -> block_pointer
    {
        const auto p = (pointer)m_allocator.allocate_one();
        const auto bp =
            block_pointer{ hierarchy_id_t::child_of(parent_id, fanout_id), p };
        m_blocks.emplace_back(bp);
        return bp;
    }

    inline auto recombine(hierarchy_id_t const& id) -> block_pointer
    {
        const auto bp =
            std::ranges::remove_if(m_blocks, [id](auto const& e) { return e.id == id; });
        const auto parent_id = hierarchy_id_t::parent_of(id);

        std::ranges::find(
            m_blocks, [parent_id](const auto e) { return e.id == parent_id; }
        );
        // TODO: Finish
        return bp;
    }

    [[nodiscard]]
    inline auto blocks() const noexcept -> container_t const&
    {
        return m_blocks;
    }

private:
    inline auto release(hierarchy_id_t const& h) noexcept
    {
        // TODO: Implement
        auto parent =
            std::ranges::remove_if(m_blocks, [h](auto const& e) { return e.id == h; });
    }

private:
    allocator_t m_allocator;
    container_t m_blocks;
};

} // namespace amr::ndt::tree

#endif // AMR_INCLUDED_NDT_TREE
