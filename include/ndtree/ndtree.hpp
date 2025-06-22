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
            int refine_flag; // 1 for refine, 2 for coarsen
        };

        pointer  ptr;
        metadata data;
    };

    struct block_pointer
    {
        using cell_metadata = typename cell_pointer::metadata;

        struct block_metadata
        {
            constexpr block_metadata(cell_metadata const& init) noexcept
                : cell_data{ utility::compile_time_utility::
                                 array_factory<cell_metadata, s_nd_fanout>(init) }
            {
            }

            [[nodiscard]]
            constexpr auto operator[](std::integral auto const i) const noexcept
                -> cell_metadata
            {
#ifdef AMR_NDTREE_CHECKBOUNDS
                assert_in_bounds(i);
#endif
                return cell_data[i];
            }

            std::array<cell_metadata, s_nd_fanout> cell_data;
        };

        constexpr block_pointer(node_index_t i, pointer p) noexcept
            : id(i)
            , ptr{ p }
            , metadata(cell_metadata{ true, 0 })
        {
        }

        node_index_t   id;
        pointer        ptr;
        block_metadata metadata;

        [[nodiscard]]
        constexpr auto operator[](std::integral auto const i) const noexcept
            -> cell_pointer
        {
#ifdef AMR_NDTREE_CHECKBOUNDS
            assert_in_bounds(i);
#endif
            return cell_pointer{ ptr + i, metadata.cell_data[i] };
        }

    public:
        constexpr auto kill_cell(std::integral auto const i) noexcept -> void
        {
#ifdef AMR_NDTREE_CHECKBOUNDS
            assert_in_bounds(i);
#endif
            assert(metadata.cell_data[i].alive);
            metadata.cell_data[i].alive = false;
        }

        constexpr auto revive_cell(std::integral auto const i) noexcept -> void
        {
#ifdef AMR_NDTREE_CHECKBOUNDS
            assert_in_bounds(i);
#endif
            assert(!metadata.cell_data[i].alive);
            metadata.cell_data[i].alive = true;
        }

        [[nodiscard]]
        constexpr auto alive_any() const noexcept -> bool
        {
            return std::ranges::any_of(
                metadata.cell_data, [](auto const& e) { return e.alive; }
            );
        }

        [[nodiscard]]
        constexpr auto alive_all() const noexcept -> bool
        {
            return std::ranges::all_of(
                metadata.cell_data, [](auto const& e) { return e.alive; }
            );
        }
        [[nodiscard]]
        constexpr auto coarsen_all() const noexcept -> bool
        {
            return std::ranges::all_of(
                metadata.cell_data, [](auto const& e) { return e.refine_flag == 2; }
            );
        }

        [[nodiscard]]
        static constexpr auto size() noexcept -> decltype(s_nd_fanout)
        {
            return s_nd_fanout;
        }

        [[nodiscard]]
        constexpr auto operator<(block_pointer const& other) const -> bool
        {
            return id < other.id;
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
        m_blocks.emplace_back(node_index_t::root(), p);
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
    auto apply_refine_coarsen()
    {
        // std::cout << "Total blocks before refinement: " << m_blocks.size() << std::endl;
        std::vector<node_index_t> to_coarsen;
        for (size_t idx = 0; idx < m_blocks.size(); ++idx) {
            auto& block = m_blocks[idx];
            if (block.coarsen_all() && block.alive_all())
            {
                // std::cout << "Servus aus der coarseing if " << std::endl;
                to_coarsen.push_back(block.id);
                // [[maybe_unused]] auto _ = recombine(block.id);
            }
            
            // std::cout << "Block " << idx << " id: " <<  block.id.id() << std::endl;

            auto [coords, level] = node_index_t::decode(block.id.id());
            // std::cout << "Block " << idx << " level: " << (int)level << " coords: " << coords[0] << "," << coords[1] << std::endl;

            if (level >= node_index_t::max_depth()) {
                // std::cout << "ERROR: Block at invalid level!" << std::endl;
                continue;
            }

            for (auto i = decltype(s_nd_fanout){}; i != s_nd_fanout; ++i) {

                block = m_blocks[idx];

                if(block.metadata.cell_data[i].refine_flag == 1 && block.metadata.cell_data[i].alive) {
                    auto child_id = node_index_t::child_of(block.id, i);
                    // std::cout << "About to fragment child at level " << (int)level + 1 << std::endl;
                    [[maybe_unused]] auto _ = fragment(child_id);
                }
            }
        }
        // Erase from highest to lowest to avoid shifting issues
        for (auto idx = to_coarsen.size(); idx-- > 0; ) {
            // std::cout << "calling recombine for " << to_coarsen[idx].id() << std::endl;
            [[maybe_unused]] auto _ = recombine(to_coarsen[idx]);
        }
    }

    template <typename Lambda>
    auto compute_refine_flag(Lambda&& condition) {
        for (auto& block : m_blocks) {
            for (auto i = decltype(s_nd_fanout){}; i != s_nd_fanout; ++i) {
                if ( block.metadata.cell_data[i].alive)
                {
                    auto child_id = node_index_t::child_of(block.id, i);
                    block.metadata.cell_data[i].refine_flag = condition(child_id);
                }
            }
        }
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
        // std::cout << "offset of parent cell " << static_cast<int>(offset) << std::endl;
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
