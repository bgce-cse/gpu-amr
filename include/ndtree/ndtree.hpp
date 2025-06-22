#ifndef AMR_INCLUDED_NDTREE
#define AMR_INCLUDED_NDTREE

#include "allocators/free_list_buffer_allocator.hpp"
#include "ndconcepts.hpp"
#include "ndhierarchy.hpp"
#include "utility/compile_time_utility.hpp"
#include "utility/constexpr_functions.hpp"
#include <algorithm>
#include <concepts>
#include <cstdint>
#include <optional>

#ifndef NDEBUG
#    define AMR_NDTREE_CHECKBOUNDS
#endif

namespace amr::ndt::tree
{

template <typename T, concepts::NodeIndex Node_Index>
class ndtree
{
public:
    using value_type                  = T;
    using pointer                     = value_type*;
    using const_pointer               = value_type const*;
    using node_index_t                = Node_Index;
    using node_index_directon_t       = typename node_index_t::direction_t;
    static constexpr auto s_nd_fanout = node_index_t::nd_fanout();

    static_assert(s_nd_fanout > 1);

    static constexpr auto s_block_size = sizeof(value_type) * s_nd_fanout;
    using allocator_t =
        allocator::free_list_buffer_allocator<s_block_size, alignof(value_type)>;

    struct cell_pointer
    {
        struct metadata
        {
            bool alive;
        };

        pointer  ptr;
        metadata data;
    };

    struct block_pointer
    {
        using cell_metadata = typename cell_pointer::metadata;

        struct block_metadata
        {
            block_metadata(cell_metadata const& init) noexcept
                : cell_data{ utility::compile_time_utility::
                                 array_factory<cell_metadata, s_nd_fanout>(init) }
            {
            }

            std::array<cell_metadata, s_nd_fanout> cell_data;
        };

        block_pointer(node_index_t i, pointer p) noexcept
            : id(i)
            , ptr{ p }
            , metadata(cell_metadata{ true })
        {
        }

        node_index_t   id;
        pointer        ptr;
        block_metadata metadata;

        [[nodiscard]]
        auto operator[](std::integral auto const i) const noexcept -> cell_pointer
        {
#ifdef AMR_NDTREE_CHECKBOUNDS
            assert_in_bounds(i);
#endif
            return cell_pointer{ ptr + i, metadata.cell_data[i] };
        }

        auto operator<(block_pointer const& other) const -> bool
        {
            return id < other.id;
        }

    public:
        auto kill_cell(std::integral auto const i) noexcept -> void
        {
#ifdef AMR_NDTREE_CHECKBOUNDS
            assert_in_bounds(i);
#endif
            assert(metadata.cell_data[i].alive);
            metadata.cell_data[i].alive = false;
        }

        auto revive_cell(std::integral auto const i) noexcept -> void
        {
#ifdef AMR_NDTREE_CHECKBOUNDS
            assert_in_bounds(i);
#endif
            assert(!metadata.cell_data[i].alive);
            metadata.cell_data[i].alive = true;
        }

        [[nodiscard]]
        auto alive_any() const noexcept -> bool
        {
            return std::ranges::any_of(
                metadata.cell_data, [](auto const& e) { return e.alive; }
            );
        }

        [[nodiscard]]
        auto alive_all() const noexcept -> bool
        {
            return std::ranges::all_of(
                metadata.cell_data, [](auto const& e) { return e.alive; }
            );
        }

    private:
#ifdef AMR_NDTREE_CHECKBOUNDS
        static auto assert_in_bounds(std::integral auto const idx) noexcept -> void
        {
            if constexpr (std::is_signed_v<decltype(idx)>)
            {
                assert(idx >= 0);
            }
            assert(idx < s_nd_fanout);
        }
#endif
    };

    using container_t                = std::vector<block_pointer>;
    using container_iterator_t       = typename container_t::iterator;
    using container_const_iterator_t = typename container_t::const_iterator;

public:
    ndtree() noexcept
        : m_allocator()
        , m_blocks()
    {
        const auto p = (pointer)m_allocator.allocate_one();
        m_blocks.emplace_back(node_index_t::zeroth_generation(), p);
    }

    [[nodiscard]]
    auto fragment(node_index_t const& node_id) -> block_pointer
    {
        auto bp = find_block(node_index_t::parent_of(node_id));
        assert(bp.has_value());

        bp.value()->kill_cell(node_index_t::offset_of(node_id));

        /*
         * TODO: Reuse this block if it is not alive anymore
        const auto alive = bp->alive();
        if (alive)
        {
        }
        else
        {

        }
          */
        const auto p      = reinterpret_cast<pointer>(m_allocator.allocate_one());
        const auto new_bp = block_pointer(node_id, p);
        m_blocks.emplace_back(new_bp);
        return new_bp;
    }

    [[nodiscard]]
    auto recombine(node_index_t const& node_id) -> cell_pointer
    {
        auto bp = find_block(node_id);
        assert(bp.has_value());

        auto pbp = find_block(node_index_t::parent_of(node_id));
        assert(pbp.has_value());

        release(bp.value());
        const auto  offset = node_index_t::offset_of(node_id);
        auto const& it     = pbp.value();
        it->revive_cell(offset);
        return it->operator[](offset);
    }

    [[nodiscard]]
    auto get_block(node_index_t const& node_id) const noexcept
        -> std::optional<block_pointer>
    {
        auto bp = find_block(node_id);
        if (!bp.has_value()) return std::nullopt;
        return *bp.value();
    }

    [[nodiscard]]
    auto get_cell(node_index_t const& node_id) const noexcept
        -> std::optional<block_pointer>
    {
        const auto parent = node_index_t::parent_of(node_id);
        const auto offset = node_index_t::offset_of(node_id);
        auto       bp     = find_block(parent);
        if (!bp.has_value()) return std::nullopt;
        return bp.value()->operator[](offset);
    }

    [[nodiscard]]
    auto blocks() const noexcept -> container_t const&
    {
        return m_blocks;
    }

private:
    [[nodiscard]]
    auto find_block(node_index_t const& node_id) const noexcept
        -> std::optional<container_const_iterator_t>
    {
        auto it = std::ranges::find_if(
            m_blocks, [&id = node_id](auto const& e) { return e.id == id; }
        );
        return it == m_blocks.end() ? std::nullopt : std::optional{ it };
    }

    [[nodiscard]]
    auto find_block(node_index_t const& node_id) noexcept
        -> std::optional<container_iterator_t>
    {
        auto it = std::ranges::find_if(
            m_blocks, [&id = node_id](auto const& e) { return e.id == id; }
        );
        return it == m_blocks.end() ? std::nullopt : std::optional{ it };
    }

    auto release(container_iterator_t const& bp) noexcept -> void
    {
        auto p = bp->ptr;
        for (auto i = decltype(s_nd_fanout){}; i != s_nd_fanout; ++i)
        {
            //(p[i]).~();
        }
        m_allocator.deallocate_one(reinterpret_cast<std::byte*>(p));
        m_blocks.erase(bp);
    }

private:
    allocator_t m_allocator;
    container_t m_blocks;
};

} // namespace amr::ndt::tree

#endif // AMR_INCLUDED_NDT_TREE
