#ifndef GPU_AMR_NDBOX
#define GPU_AMR_NDBOX

#include "utility/error_handling.hpp"
#include "logging.hpp"
#include "ndboundary.hpp"
#include "ndt_concepts.hpp"
#include "utility/constexpr_functions.hpp"
#include <cmath>
#include <cstddef>
#include <vector>

#define DEBUG_LOG_NDTREE (true)

namespace ndt
{

template <std::size_t Fanout, ndt_concepts::cell Cell_Type>
class NDBox
{
public:
    using cell_t                             = Cell_Type;
    using point_t                            = typename cell_t::position_t;
    using value_type                         = typename cell_t::value_type;
    using box_t                              = NDBox<Fanout, cell_t>;
    using boundary_t                         = NDBoundary<point_t>;
    using depth_t                            = unsigned int;
    inline static constexpr auto s_dimension = point_t::s_dimension;
    inline static constexpr auto s_fanout    = Fanout;
    inline static constexpr auto s_subdivisions =
        utility::cx_functions::pow(s_fanout, s_dimension);

public:
    NDBox(boundary_t boundary, depth_t depth, depth_t max_depth, box_t* parent)
        : m_boundary{ boundary }
        , m_parent{ parent }
        , m_depth{ depth }
    {
    }

    [[nodiscard]]
    auto fragmented() const noexcept -> bool
    {
        return m_fragmented;
    }

    auto print_info(std::ostream& os) const -> void
    {
        static auto header = [](auto depth)
        {
            return std::string(depth, '\t');
        };
        os << header(m_depth) << "<ndbox<" << s_dimension << ">>\n ";
        os << header(m_depth + 1) << "Boundary: " << m_boundary << '\n';
        os << header(m_depth + 1) << "Depth " << m_depth << '\n';
        os << header(m_depth + 1) << "Fragmented: " << m_fragmented << '\n';
        if (!m_fragmented)
        {
            for (auto const& e : contained_elements())
            {
                os << header(m_depth + 1) << e->repr() << '\n';
            }
        }
        else
        {
            for (auto const& b : subboxes())
            {
                b.print_info(os);
            }
        }
        os << header(m_depth) << "<\\ndbox<" << s_dimension << ">>\n";
    }

    [[nodiscard]]
    auto boxes() const -> std::size_t
    {
        return m_fragmented ? std::ranges::fold_left(
                                  subboxes(),
                                  std::ranges::size(subboxes()),
                                  [](auto acc, const auto& b) { return acc + b.boxes(); }
                              )
                            : 0uz;
    }

    [[nodiscard]]
    auto elements() const -> std::size_t
    {
        return m_fragmented
                   ? std::ranges::fold_left(
                         subboxes(),
                         0uz,
                         [](auto acc, const auto& e) { return acc + e.elements(); }
                     )
                   : std::ranges::size(contained_elements());
    }

#if __GNUC__ >= 14
    [[nodiscard]]
    auto contained_elements(this auto&& self) noexcept -> auto&&
    {
        std::forward<decltype(self)>(self).assert_not_fragmented();
        return std::get<0>(std::forward<decltype(self)>(self).m_elements);
    }

    [[nodiscard]]
    auto subboxes(this auto&& self) noexcept -> auto&&
    {
        std::forward<decltype(self)>(self).assert_fragmented();
        return std::get<1>(std::forward<decltype(self)>(self).m_elements);
    }
#else
    [[nodiscard]]
    auto contained_elements() noexcept -> auto&
    {
        assert_not_fragmented();
        return std::get<0>(m_elements);
    }

    [[nodiscard]]
    auto contained_elements() const noexcept -> auto const&
    {
        assert_not_fragmented();
        return std::get<0>(m_elements);
    }

    [[nodiscard]]
    auto subboxes() noexcept -> auto&
    {
        assert_fragmented();
        return std::get<1>(m_elements);
    }

    [[nodiscard]]
    auto subboxes() const noexcept -> auto const&
    {
        assert_fragmented();
        return std::get<1>(m_elements);
    }
#endif

private:
    // because you cannot portably have a macro expansion (assert) inside #if
    // #endif
    inline auto assert_fragmented() const noexcept -> void
    {
        assert(fragmented());
    }

    inline auto assert_not_fragmented() const noexcept -> void
    {
        assert(!fragmented());
    }

    auto fragment() noexcept -> void
    {
#if DEBUG_LOG_NDTREE
        utility::logging::default_source::log(
            utility::logging::severity_level::info, "Fragentation started"
        );
#endif
        using size_type = decltype(s_dimension);
        if (m_fragmented)
        {
            return;
        }
        m_elements   = std::vector<box_t>();
        m_fragmented = true;
        subboxes().reserve(s_subdivisions);
        if constexpr (s_fanout == 2)
        {
            static_assert(
                s_fanout == 2,
                "The divisions are only binary with implementation! It could be "
                "generalzed "
                "if needed tho..."
            );
            for (auto binary_div = decltype(s_subdivisions){ 0 };
                 binary_div != s_subdivisions;
                 ++binary_div)
            {
                point_t min;
                point_t max;
                for (auto i = decltype(s_dimension){ 0 }; i != s_dimension; ++i)
                {
                    const auto top_half = (binary_div & (1 << i)) > 0;
                    min[i] = top_half ? m_boundary.mid(i) : m_boundary.min(i);
                    max[i] = top_half ? m_boundary.max(i) : m_boundary.mid(i);
                }
                subboxes().push_back(
                    box_t{
                        boundary_t{ min, max },
                        m_depth + 1, m_max_depth, this
                }
                );
            }
        }
        else
        {
            for (auto n = decltype(s_subdivisions){}; n != s_subdivisions; ++n)
            {
                point_t min;
                point_t max;

                auto dim_idx = static_cast<value_type>(n);

                for (auto i = size_type{}; i != s_dimension; ++i)
                {
                    const auto j = static_cast<value_type>(
                        static_cast<size_type>(dim_idx) % s_fanout
                    );
                    const auto delta = (m_boundary.max(i) - m_boundary.min(i)) /
                                       static_cast<value_type>(s_fanout);
                    min[i]  = m_boundary.min(i) + j * delta;
                    max[i]  = m_boundary.min(i) + (j + value_type{ 1 }) * delta;
                    dim_idx = std::floor(dim_idx / static_cast<value_type>(s_fanout));
                }
                subboxes().push_back(
                    box_t{
                        boundary_t{ min, max },
                        m_depth + 1, m_max_depth, this
                }
                );
            }
        }
        for (auto const* const s : contained_elements())
        {
            if (!s)
            {
                continue;
            }
            for (auto&& b : subboxes())
            {
                if (const auto success = b.insert(s); success)
                {
                    break;
                }
            }
        }
#if DEBUG_LOG_NDTREE
        utility::logging::default_source::log(
            utility::logging::severity_level::info, "Fragentation finished"
        );
#endif
    }

private:
    boundary_t                                            m_boundary;
    std::variant<std::vector<cell_t>, std::vector<box_t>> m_elements;
    box_t*                                                m_parent;
    std::size_t                                           m_capacity;
    bool                                                  m_fragmented = false;
    depth_t                                               m_max_depth;
    depth_t                                               m_depth;
};

} // namespace ndt

#endif // GPU_AMR_NDBOX
