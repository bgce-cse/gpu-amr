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
            int  refine_flag; // 1 for refine, 2 for coarsen
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
            constexpr auto operator[](std::integral auto const i
            ) const noexcept -> cell_metadata
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
        constexpr auto operator[](std::integral auto const i
        ) const noexcept -> cell_pointer
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

    auto fragment(std::vector<node_index_t>& to_refine)
    {
        std::cout << "[fragment] to_refine vector contains " << to_refine.size()
                  << " entries:\n";
        for (size_t i = 0; i < to_refine.size(); ++i)
        {
            std::cout << "  [" << i << "] id = " << to_refine[i].id() << std::endl;
        }
        for (size_t idx = 0; idx < to_refine.size(); idx++)
        {
            [[maybe_unused]]
            auto _ = fragment(to_refine[idx]);
        }
    }

    auto test_get_neighbor()
    {
        // iterate over all block and return neighbors of all cells
        for (size_t idx = 0; idx < m_blocks.size(); ++idx)
        {
            auto& block = m_blocks[idx];

            for (int i = 0; i < 4; i++)
            {
                if (block.metadata.cell_data[i].alive)
                {
                    auto cell_id = node_index_t::child_of(block.id, i);
                    auto result  = get_neighbors(cell_id, node_index_directon_t::right);
                    std::cout << "I am cell " << cell_id.id()
                              << " and my neighbors are :  " << std::endl;
                    if (result)
                    {
                        // result is a pair: {neighbor_id, std::vector<offsets>}
                        auto [neighbor_id, offsets] = *result;
                        std::cout << "Neighbor block id: " << neighbor_id.id()
                                  << std::endl;
                        for (auto offset : offsets)
                        {
                            std::cout << "Neighbor cell offset: " << offset << std::endl;
                        }
                    }
                    else
                    {
                        std::cout << "No neighbor in that direction." << std::endl;
                    }
                }
            }
        }
    }

    auto get_neighbors(node_index_t const& node_id, node_index_directon_t dir)
        -> std::optional<
            std::pair<node_index_t, std::vector<typename node_index_t::offset_t>>>
    {
        // std::cout << "[get_neighbors] Called for cell id: " << node_id.id()
        //           << " in direction: " << static_cast<int>(dir) << std::endl;

        auto parent_id = node_index_t::parent_of(node_id.id());
        // std::cout << "[get_neighbors] Parent id: " << parent_id.id() << std::endl;

        auto bp_node = find_block(parent_id);
        if (!bp_node.has_value())
        {
            // std::cout << "[get_neighbors] Parent block not found!" << std::endl;
            assert(false);
        }

        auto offset = node_index_t::offset_of(node_id.id());
        // std::cout << "[get_neighbors] Offset in parent: " << offset << std::endl;

        if (!bp_node.value()->metadata.cell_data[offset].alive)
        {
            // std::cout << "[get_neighbors] Cell is not alive in parent block!"
            //           << std::endl;
                      assert(false);
            return std::nullopt;
        }

        std::vector<typename node_index_t::offset_t> index_vector;
        auto direct_neighbor = node_index_t::neighbour_at(node_id, dir);

        if (!direct_neighbor)
        {
            // std::cout << "[get_neighbors] No direct neighbor exists in that direction."
                    //   << std::endl;
            return std::nullopt;
        }
        // std::cout << "[get_neighbors] Direct neighbor id: "
        //           << direct_neighbor.value().id() << std::endl;

        auto d_neighbor = find_block(direct_neighbor.value());
        if (d_neighbor.has_value())
        {
            // std::cout << "[get_neighbors] Direct neighbor block found!" << std::endl;
            typename node_index_t::offset_t offset0, offset1;
            switch (dir)
            {
                case node_index_directon_t::left:
                    offset0 = 1;
                    offset1 = 3;
                    break;
                case node_index_directon_t::right:
                    offset0 = 0;
                    offset1 = 2;
                    break;
                case node_index_directon_t::bottom:
                    offset0 = 2;
                    offset1 = 3;
                    break;
                case node_index_directon_t::top:
                    offset0 = 0;
                    offset1 = 1;
                    break;
                default: break;
            }
            // std::cout << "[get_neighbors] Neighbor cell offsets: " << offset0 << ", "
            //           << offset1 << std::endl;
            index_vector.push_back(offset0);
            index_vector.push_back(offset1);
            return std::make_optional(
                std::make_pair(direct_neighbor.value(), index_vector)
            );
        }

        auto neighbor_parent = node_index_t::parent_of(direct_neighbor.value());
        // std::cout << "[get_neighbors] Checking neighbor's parent id: "
        //           << neighbor_parent.id() << std::endl;
        auto p_neighbor = find_block(neighbor_parent);
        if (p_neighbor.has_value())
        {
            // std::cout << "[get_neighbors] Parent of direct neighbor found!" << std::endl;
            auto neighbor_offset = node_index_t::offset_of(direct_neighbor.value());
            // std::cout << "[get_neighbors] Offset in neighbor's parent: "
            //           << neighbor_offset << std::endl;
            index_vector.push_back(neighbor_offset);
            return std::make_optional(std::make_pair(neighbor_parent, index_vector));
        }

        auto neighbor_grandparent = node_index_t::parent_of(neighbor_parent);
        // std::cout << "[get_neighbors] Checking neighbor's grandparent id: "
        //           << neighbor_grandparent.id() << std::endl;
        auto gp_neighbor = find_block(neighbor_grandparent);
        if (gp_neighbor.has_value())
        {
            // std::cout << "[get_neighbors] Grandparent of direct neighbor found!"
            //           << std::endl;
            auto neighbor_parent_offset = node_index_t::offset_of(neighbor_parent);
            // std::cout << "[get_neighbors] Offset in neighbor's grandparent: "
            //           << neighbor_parent_offset << std::endl;
            index_vector.push_back(neighbor_parent_offset);
            return std::make_optional(std::make_pair(neighbor_grandparent, index_vector));
        }

        // std::cout << "[get_neighbors] No neighbor found after all checks!" << std::endl;
        assert(false);
    }

    [[nodiscard]]
    auto apply_refine_coarsen()
        -> std::pair<std::vector<node_index_t>, std::vector<node_index_t>>

    // supposed to return vecitd of cells to refine and vecotr fo block to coarsen, THis
    // is then given t another fucntion ensuring the balancing.
    {
        // std::cout << "Total blocks before refinement: " << m_blocks.size() <<
        // std::endl;
        std::vector<node_index_t> to_coarsen;
        std::vector<node_index_t> to_refine;
        for (size_t idx = 0; idx < m_blocks.size(); ++idx)
        {
            auto& block = m_blocks[idx];
            if (block.coarsen_all() && block.alive_all())
            {
                // std::cout << "Servus aus der coarseing if " << std::endl;
                to_coarsen.push_back(block.id);
                // [[maybe_unused]] auto _ = recombine(block.id);
            }

            // std::cout << "Block " << idx << " id: " <<  block.id.id() << std::endl;

            auto [coords, level] = node_index_t::decode(block.id.id());
            // std::cout << "Block " << idx << " level: " << (int)level << " coords: " <<
            // coords[0] << "," << coords[1] << std::endl;

            if (level >= node_index_t::max_depth())
            {
                // std::cout << "ERROR: Block at invalid level!" << std::endl;
                continue;
            }

            for (auto i = decltype(s_nd_fanout){}; i != s_nd_fanout; ++i)
            {
                block = m_blocks[idx];

                if (block.metadata.cell_data[i].refine_flag == 1 &&
                    block.metadata.cell_data[i].alive)
                {
                    auto child_id = node_index_t::child_of(block.id, i);
                    to_refine.push_back(child_id.id());
                    // std::cout << "About to fragment child at level " << (int)level + 1
                    // << std::endl;
                    // [[maybe_unused]]
                    // auto _ = fragment(child_id);
                }
            }
        }
        return { to_refine, to_coarsen };

        // Erase from highest to lowest to avoid shifting issues
        // for (auto idx = to_coarsen.size(); idx-- > 0;)
        // {
        //     // std::cout << "calling recombine for " << to_coarsen[idx].id() <<
        //     std::endl;
        //     [[maybe_unused]]
        //     auto _ = recombine(to_coarsen[idx]);
        // }
    }

    auto balancing(
        std::vector<node_index_t>& to_refine,
        std::vector<node_index_t>& to_coarsen
    )
    {
        // check if balancing condition is violated.
        // first refinement
        constexpr node_index_directon_t directions[] = { node_index_directon_t::left,
                                                         node_index_directon_t::right,
                                                         node_index_directon_t::top,
                                                         node_index_directon_t::bottom };

        for (size_t i = 0; i < to_refine.size(); i++)
        {
            auto cell_id         = to_refine[i];
            auto [coords, level] = node_index_t::decode(cell_id.id());
            std::cout << "[balancing] Refinement  Checking cell " << cell_id.id()
                      << " at level " << (int)level << " coords: (" << coords[0] << ","
                      << coords[1] << ")\n";
            for (auto direction : directions)
            {
                auto result = get_neighbors(cell_id.id(), direction);
                if (!result)
                {
                    // std::cout << "  No neighbor in direction "
                    //           << static_cast<int>(direction) << "\n";
                    continue;
                }
                auto [neighbor_id, offsets]  = *result;
                auto [__, level_bp_neighbor] = node_index_t::decode(neighbor_id.id());
                // std::cout << "  Neighbor in direction " << static_cast<int>(direction)
                //           << ": id=" << neighbor_id.id()
                //           << " bp level=" << (int)level_bp_neighbor << " offsets: ";
                // for (auto off : offsets)
                //     std::cout << off << " ";
                // std::cout << "\n";
                if (level_bp_neighbor < level - 1)
                {
                    std::cout
                        << "    [balancing] Balancing violation! Refining neighbor cell "

                        << node_index_t::child_of(neighbor_id, offsets[0]).id() << "\n";
                    // balancing condition violated
                    // push this cell to refinement cells (it will be checked itself later
                    // as it is appended at the end of the vector)
                    auto new_id = node_index_t::child_of(neighbor_id, offsets[0]).id();
                    if (std::find_if(
                            to_refine.begin(),
                            to_refine.end(),
                            [&](const node_index_t& n) { return n.id() == new_id; }
                        ) == to_refine.end())
                    {
                        to_refine.push_back(
                            node_index_t::child_of(neighbor_id, offsets[0])
                        );
                    }
                }
            }
        }
        std::vector<node_index_t> parents_of_to_refine;
        for (auto child_cell : to_refine)
        {
            auto parent_block = node_index_t::parent_of(child_cell.id());
            parents_of_to_refine.push_back(parent_block.id());
        }

        // check coarsening after refining - if there is a conflict refinment wins and the
        // cell needs to be removed from coarsening
        for (size_t i = 0; i < to_coarsen.size(); i++)
        {
            auto block_id        = to_coarsen[i];
            auto [coords, level] = node_index_t::decode(block_id.id());
            std::cout << "[balancing] Coarsening  Checking block " << block_id.id()
                      << " at level " << (int)level << " coords: (" << coords[0] << ","
                      << coords[1] << ")\n";

            for (auto direction : directions)
            {
                auto offset0 = 0;
                auto offset1 = 0;

                switch (direction)
                {
                    case node_index_directon_t::left:
                        offset0 = 0;
                        offset1 = 2;
                        break;
                    case node_index_directon_t::right:
                        offset0 = 1;
                        offset1 = 3;
                        break;
                    case node_index_directon_t::bottom:
                        offset0 = 2;
                        offset1 = 3;
                        break;
                    case node_index_directon_t::top:
                        offset0 = 0;
                        offset1 = 1;
                        break;
                    default: break;
                }
                std::vector<int> offsets;
                offsets.push_back(offset0);
                offsets.push_back(offset1);
                for (auto offset : offsets)
                {
                    auto child_cell = node_index_t::child_of(block_id.id(), offset);
                    auto result     = get_neighbors(child_cell.id(), direction);
                    if (!result)
                    {
                        // std::cout << "  No neighbor in direction "
                        //           << static_cast<int>(direction) << "\n";
                        continue;
                    }
                    auto [neighbor_id, neighbor_offsets]  = *result;
                    auto [__, level_bp_neighbor] = node_index_t::decode(neighbor_id.id());
                    // std::cout << "  Neighbor in direction " << static_cast<int>(direction)
                    //           << ": id=" << neighbor_id.id()
                    //           << " bp level=" << (int)level_bp_neighbor << " offsets: ";
                    // for (auto off : neighbor_offsets)
                    //     std::cout << off << " ";
                    // std::cout << "\n";
                    if (level_bp_neighbor > level)
                    {
                        // std::cout << "    [balancing] Balancing violation! coarsening  "
                        //              "neighbor cells " <<std::endl;

                        // balancing condition violated
                        // push this cell to coarsening cells (it will be checked itself
                        // later as it is appended at the end of the vector)

                        if (std::find_if(
                                to_coarsen.begin(),
                                to_coarsen.end(),
                                [&](const node_index_t& n)
                                { return n.id() == neighbor_id.id(); }
                            ) == to_coarsen.end() &&
                            std::find_if(
                                parents_of_to_refine.begin(),
                                parents_of_to_refine.end(),
                                [&](const node_index_t& n)
                                { return n.id() == neighbor_id.id(); }
                            ) == parents_of_to_refine.end())
                        {
                            to_coarsen.push_back(neighbor_id);
                            
                        }
                    }
                }
            }
        }
    }

    template <typename Lambda>
    auto compute_refine_flag(Lambda&& condition)
    {
        for (auto& block : m_blocks)
        {
            for (auto i = decltype(s_nd_fanout){}; i != s_nd_fanout; ++i)
            {
                if (block.metadata.cell_data[i].alive)
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
        const auto offset = node_index_t::offset_of(node_id);
        // std::cout << "offset of parent cell " << static_cast<int>(offset) << std::endl;
        auto const& it = pbp.value();
        it->revive_cell(offset);
        return it->operator[](offset);
    }

    auto recombine(const std::vector<node_index_t>& node_ids) -> void
    {
        std::cout << "[recombine] to_coarsen vector contains " << node_ids.size()
                  << " entries:\n";
        for (size_t i = 0; i < node_ids.size(); ++i)
        {
            auto [coords, level] = node_index_t::decode(node_ids[i].id());
            std::cout << "  [" << i << "] id = " << node_ids[i].id()
                      << " coords: (" << coords[0] << "," << coords[1] << ")"
                      << " level: " << static_cast<int>(level) << std::endl;
        }

        for (const auto& node_id : node_ids)
        {
            auto [coords, level] = node_index_t::decode(node_id.id());
            std::cout << "[recombine] Recombining node_id: " << node_id.id()
                      << " coords: (" << coords[0] << "," << coords[1] << ")"
                      << " level: " << static_cast<int>(level) << std::endl;
            [[maybe_unused]]
            auto _ = recombine(node_id);
        }
    }

    [[nodiscard]]
    auto get_block(node_index_t const& node_id
    ) const noexcept -> std::optional<block_pointer>
    {
        auto bp = find_block(node_id);
        if (!bp.has_value()) return std::nullopt;
        return *bp.value();
    }

    [[nodiscard]]
    auto get_cell(node_index_t const& node_id
    ) const noexcept -> std::optional<block_pointer>
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
    auto find_block(node_index_t const& node_id
    ) const noexcept -> std::optional<container_const_iterator_t>
    {
        auto it = std::ranges::find_if(
            m_blocks, [&id = node_id](auto const& e) { return e.id == id; }
        );
        return it == m_blocks.end() ? std::nullopt : std::optional{ it };
    }

    [[nodiscard]]
    auto find_block(node_index_t const& node_id
    ) noexcept -> std::optional<container_iterator_t>
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
