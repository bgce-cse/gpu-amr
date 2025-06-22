#ifndef GPU_AMR_NDTREE
#define GPU_AMR_NDTREE

#include "ndboundary.hpp"
#include "ndbox.hpp"
#include "ndt_concepts.hpp"
#include <cassert>
#include <iterator>
#include <optional>
#include <ostream>
#include <ranges>

namespace ndt
{

template <std::size_t Fanout, ndt_concepts::cell Cell_Type>
    requires(Fanout > 1 && Cell_Type::position_t::s_dimension > 0)
class NDTree
{
public:
    using cell_t                                = Cell_Type;
    using position_t                            = typename cell_t::position_t;
    using value_type                            = typename cell_t::value_type;
    using size_type                             = std::size_t;
    using box_t                                 = NDBox<Fanout, cell_t>;
    using depth_t                               = typename box_t::depth_t;
    using point_t                               = typename cell_t::position_t;
    using boundary_t                            = NDBoundary<point_t>;
    inline static constexpr auto s_dimension    = cell_t::s_dimension;
    inline static constexpr auto s_fanout       = box_t::s_fanout;
    inline static constexpr auto s_subdivisions = box_t::s_subdivisions;

public:
    NDTree(boundary_t limits, depth_t const max_depth),
        m_box(limits, 0uz, max_depth, nullptr), m_max_depth{ max_depth }
    {
        for (auto const& e : collection)
        {
            [[maybe_unused]]
            const auto inserted = insert(&e);
            assert(inserted);
        }
    }

    [[nodiscard]]
    inline auto insert(cell_t const* const sp) noexcept -> bool
    {
        return m_box.insert(sp);
    }

    inline auto reorganize() noexcept -> void
    {
        m_box.reorganize();
    }

    [[nodiscard]]
    constexpr auto size() const noexcept -> size_type
    {
        return std::ranges::size(m_data_view);
    }

    auto print_info(std::ostream& os = std::cout) const -> void
    {
        os << "<ndtree <" << s_fanout << ", " << s_dimension << ">>\n";
        os << "Capacity: " << m_capacity << '\n';
        os << "Max depth: " << m_max_depth << '\n';
        os << "Elements: " << m_box.elements() << " out of "
           << std::ranges::size(m_data_view) << '\n';
        os << "<\\ndtree<" << s_fanout << ", " << s_dimension << ">>\n";
    }

    [[nodiscard]]
    inline auto box() const -> box_t const&
    {
        return m_box;
    }

private:
    box_t     m_box;
    depth_t   m_max_depth;
    size_type m_capacity;
};

template <std::size_t Fanout, ndt_concepts::cell Sample_Type>
auto operator<<(std::ostream& os, NDTree<Fanout, Sample_Type> const& tree)
    -> std::ostream&
{
    tree.print_info(os);
    tree.box().print_info(os);
    return os;
}

} // namespace ndt

#endif // GPU_AMR_NDTREE
