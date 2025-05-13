#pragma once

#include "constexpr_functions.hpp"
#include "error_handling.hpp"
#include "logging.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iterator>
#include <numeric>
#include <optional>
#include <ostream>
#include <ranges>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#ifdef DEBUG_NDTREE
#include <iostream>
#endif

// I think this is actually not needed anymore but do not use this recreational
// code for such a high dimensional space
#define NDTREE_MAX_DIMENSIONS                                                            \
    64 // Because of how binary boundary subdivisions are computed

namespace ndt
{

namespace concepts
{
template <typename T>
concept Point = requires(T t) {
    typename T::value_type;
    T::s_dimension;
    t[0];
    std::begin(t);
    std::end(t);
};

template <typename T>
concept boundary_concept = requires(T t) {
    t.min();
    t.max();
    t.mid(0);
};

template <typename T>
concept sample_concept = requires(T t) {
    typename T::value_type;
    T::s_dimension;
    { t.position() } -> std::convertible_to<typename T::position_t>;
    t.properties();
    { merge(std::array{ t, t }) } -> std::same_as<std::optional<T>>;
} && std::is_destructible_v<T>;

} // namespace concepts

template <concepts::Point Point_Type>
class ndboundary
{
public:
    using point_t                            = Point_Type;
    using value_type                         = typename point_t::value_type;
    inline static constexpr auto s_dimension = point_t::s_dimension;
    using index_t                            = std::remove_const_t<decltype(s_dimension)>;

public:
    constexpr ndboundary() noexcept = delete;

    constexpr ndboundary(point_t const& p1, point_t const& p2) noexcept
    {
        for (index_t i = 0; i != s_dimension; ++i)
        {
            // Explicit copy to avoid dangling reference to a temporary
            const std::pair<value_type, value_type> r = std::minmax(p1[i], p2[i]);
            m_min[i]                                  = r.first;
            m_max[i]                                  = r.second;
        }
    }

    constexpr ndboundary(value_type const& v1, value_type v2) noexcept
    {
        const std::pair<value_type, value_type> r = std::minmax(v1, v2);
        for (index_t i = 0; i != s_dimension; ++i)
        {
            m_min[i] = r.first;
            m_max[i] = r.second;
        }
    }

    [[nodiscard]]
    constexpr auto min() const -> point_t const&
    {
        return m_min;
    }

    [[nodiscard]]
    constexpr auto max() const -> point_t const&
    {
        return m_max;
    }

    [[nodiscard]]
    constexpr auto min(index_t const idx) const -> value_type
    {
        return m_min[idx];
    }

    [[nodiscard]]
    constexpr auto max(index_t const idx) const -> value_type
    {
        return m_max[idx];
    }

    [[nodiscard]]
    constexpr auto mid(index_t const idx) const -> value_type
    {
        return std::midpoint(m_max[idx], m_min[idx]);
    }

    [[nodiscard]]
    constexpr auto operator<=>(ndboundary const&) const = default;

private:
    point_t m_min;
    point_t m_max;
};

template <concepts::Point Point_Type>
auto operator<<(std::ostream& os, ndboundary<Point_Type> const& b) -> std::ostream&
{
    os << "{ " << b.min() << " }, { " << b.max() << " }";
    return os;
}

namespace detail
{

template <concepts::Point Point_Type>
[[nodiscard]]
constexpr auto in(
    Point_Type const&               p,
    ndboundary<Point_Type> const&   b,
    typename Point_Type::value_type tol
) noexcept -> bool
{
    assert(tol > 0);
    for (auto i = decltype(Point_Type::s_dimension){}; i != Point_Type::s_dimension; ++i)
    {
        if (p[i] < (b.min(i) - tol) || p[i] > (b.max(i) + tol))
        {
            return false;
        }
    }
    return true;
}

template <concepts::Point Point_Type>
[[nodiscard]]
constexpr auto count_in(
    std::ranges::range auto const& collection,
    ndboundary<Point_Type> const&  b
) noexcept -> std::size_t
{
    return std::ranges::count_if(collection, [b](auto const& p) { return in(p, b); });
}

[[nodiscard]]
auto compute_limits(std::ranges::range auto const& data) noexcept
    requires concepts::sample_concept<std::ranges::range_value_t<decltype(data)>>
{
    using sample_t   = std::ranges::range_value_t<decltype(data)>;
    using point_t    = typename sample_t::position_t;
    constexpr auto N = point_t::s_dimension;
    point_t        min;
    point_t        max;
    for (auto i = decltype(N){ 0 }; i != N; ++i)
    {
        const auto bounds =
            std::ranges::minmax(data | std::views::transform([i](auto const& p) -> auto {
                                    return p.position()[i];
                                }));
        min[i] = bounds.min;
        max[i] = bounds.max;
    }
    return ndboundary<point_t>{ min, max };
}

} // namespace detail

template <std::size_t Fanout, concepts::sample_concept Sample_Type>
class ndbox
{
public:
    using sample_t                           = Sample_Type;
    using point_t                            = typename sample_t::position_t;
    using box_t                              = ndbox<Fanout, sample_t>;
    inline static constexpr auto s_dimension = point_t::s_dimension;
    using value_type                         = typename sample_t::value_type;
    using boundary_t                         = ndboundary<point_t>;
    using depth_t                            = unsigned int;
    inline static constexpr auto s_fanout    = Fanout;
    inline static constexpr auto s_subdivisions =
        utility::cx_functions::pow(s_fanout, s_dimension);
    inline static constexpr auto s_boundary_tol = value_type{ 1e-4 };

public:
    ndbox(
        boundary_t  boundary,
        std::size_t max_elements,
        depth_t     depth,
        depth_t     max_depth,
        ndbox*      parent
    ) :
        m_boundary{ boundary },
        m_elements{},
        m_summary{ std::nullopt },
        m_parent{ parent },
        m_capacity{ max_elements },
        m_max_depth{ max_depth },
        m_depth{ depth }
    {
    }

    [[nodiscard]]
    auto insert(sample_t const* const sp) noexcept -> bool
    {
        if (!detail::in(sp->position(), m_boundary, s_boundary_tol))
        {
            return false;
        }
        if (!fragmented())
        {
            auto&& elements = contained_elements();
            if (elements.size() < m_capacity || m_depth == m_max_depth)
            {
                elements.push_back(sp);
#if DEBUG_NDTREE
                std::cout << "Value at " << sp->position() << " stored in Box at depth "
                          << m_depth << " with bounds " << m_boundary << '\n';
#endif
                return true;
            }
            else
            {
#if DEBUG_NDTREE
                utility::logging::default_source::log(
                    utility::logging::severity_level::info, "Fragentation started"
                );
#endif
                fragment();
#if DEBUG_NDTREE
                utility::logging::default_source::log(
                    utility::logging::severity_level::info, "Fragentation finished"
                );
#endif
            }
        }
        if (fragmented())
        {
            for (auto&& b : subboxes())
            {
                if (const auto success = b.insert(sp); success)
                {
                    return true;
                }
            }
            return false;
        }
        utility::error_handling::assert_unreachable();
    }

    auto reorganize() noexcept -> void
    {
        if (fragmented())
        {
            for (auto&& b : subboxes())
            {
                b.reorganize();
            }
        }
        else
        {
#if DEBUG_NDTREE
            const bool flag =
                std::ranges::any_of(contained_elements(), [this](auto const& e) {
                    return !detail::in(e->position(), m_boundary, s_boundary_tol);
                });
            if (flag)
            {
                std::cout << "Bounds: " << m_boundary << '\n';
                for (auto p : contained_elements())
                {
                    std::cout << p->repr() << std::endl;
                }
            }
#endif
            const auto out_of_bounds_range =
                std::ranges::partition(contained_elements(), [this](auto const* const p) {
                    return detail::in(p->position(), m_boundary, s_boundary_tol);
                });
#if DEBUG_NDTREE
            if (flag)
            {
                std::cout << "Bounds: " << m_boundary << '\n';
                for (auto p : contained_elements())
                {
                    std::cout << p->repr() << std::endl;
                }
            }
#endif
            if (out_of_bounds_range.begin() != out_of_bounds_range.end())
            {
                if (m_parent)
                {
                    for (auto const* const p : out_of_bounds_range)
                    {
                        assert(!detail::in(p->position(), m_boundary, s_boundary_tol));
                        m_parent->relocate(p);
                    }
                }
                contained_elements().erase(
                    out_of_bounds_range.begin(), out_of_bounds_range.end()
                );
            }
        }
    }

    auto relocate(sample_t const* const sp) noexcept -> void
    {
        if (!detail::in(sp->position(), m_boundary, s_boundary_tol))
        {
            if (m_parent)
            {
                m_parent->relocate(sp);
            }
#if DEBUG_NDTREE
            else
            {
                std::ostringstream ss;
                ss << "Sample at " << sp->position()
                   << " no longer tracked by the tree...";
                utility::logging::default_source::log(utility::logging::info, ss.str());
            }
#endif
        }
        else
        {
            [[maybe_unused]]
            const auto inserted = insert(sp);
            assert(inserted);
        }
    }

    auto cache_summary() noexcept -> void
    {
        if (fragmented())
        {
            for (auto&& b : subboxes())
            {
                b.cache_summary();
            }
            m_summary = merge(
                subboxes() | std::views::filter([](auto const& b) {
                    return b.summary().has_value();
                }) |
                std::views::transform([](auto const& b) { return b.summary().value(); })
            );
        }

        else
        {
            m_summary =
                merge(contained_elements() | std::views::transform([](auto const& b) {
                          return *b;
                      }));
        }
    }

    [[nodiscard]]
    inline auto fragmented() const noexcept -> bool
    {
        return m_fragmented;
    }

    [[nodiscard]]
    auto summary() const noexcept -> std::optional<sample_t> const&
    {
        return m_summary;
    }

    [[nodiscard]]
    auto diagonal_length() const noexcept -> auto
    {
        return m_boundary.diagonal_length();
    }

    auto print_info(std::ostream& os) const -> void
    {
        static auto header = [](auto depth) { return std::string(depth, '\t'); };
        os << header(m_depth) << "<ndbox<" << s_dimension << ">>\n ";
        os << header(m_depth + 1) << "Boundary: " << m_boundary << '\n';
        os << header(m_depth + 1) << "Capacity " << m_capacity << '\n';
        os << header(m_depth + 1) << "Depth " << m_depth << '\n';
        os << header(m_depth + 1) << "Fragmented: " << m_fragmented << '\n';
        os << header(m_depth + 1) << "Boxes: " << boxes() << '\n';
        if (summary().has_value())
        {
            os << header(m_depth + 1) << "Summary: " << summary().value().repr() << '\n';
        }
        os << header(m_depth + 1) << "Elements: " << elements() << '\n';
        if (!m_fragmented)
        {
            auto&& elements = contained_elements();
            for (auto const& e : elements)
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
                            : 0;
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
        using size_type = decltype(s_dimension);
        if (m_fragmented)
        {
            return;
        }
        auto samples = std::move(contained_elements());
        m_elements   = std::vector<ndbox>();
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
                subboxes().push_back(box_t{
                    boundary_t{ min, max }, m_capacity, m_depth + 1, m_max_depth, this });
            }
        }
        else
        {
            for (auto n = decltype(s_subdivisions){ 0 }; n != s_subdivisions; ++n)
            {
                point_t min;
                point_t max;

                auto dim_idx = static_cast<value_type>(n);

                for (auto i = size_type{ 0 }; i != s_dimension; ++i)
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
                subboxes().push_back(box_t{
                    boundary_t{ min, max }, m_capacity, m_depth + 1, m_max_depth, this });
            }
        }
        for (auto const* const s : samples)
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
    }

private:
    boundary_t m_boundary;

    std::variant<std::vector<sample_t const*>, std::vector<ndbox>> m_elements;
    std::optional<sample_t>                                        m_summary;
    ndbox*                                                         m_parent;
    std::size_t                                                    m_capacity;
    bool                                                           m_fragmented = false;
    depth_t                                                        m_max_depth;
    depth_t                                                        m_depth;
};

template <std::size_t Fanout, concepts::sample_concept Sample_Type>
    requires(
        Fanout > 1 && Sample_Type::position_t::s_dimension > 0 &&
        Sample_Type::position_t::s_dimension < NDTREE_MAX_DIMENSIONS
    )
class ndtree
{
public:
    using sample_t                              = Sample_Type;
    using position_t                            = typename sample_t::position_t;
    using value_type                            = typename sample_t::value_type;
    using size_type                             = std::size_t;
    inline static constexpr auto s_dimension    = sample_t::s_dimension;
    using box_t                                 = ndbox<Fanout, sample_t>;
    using depth_t                               = typename box_t::depth_t;
    using point_t                               = typename sample_t::position_t;
    using boundary_t                            = ndboundary<point_t>;
    inline static constexpr auto s_fanout       = box_t::s_fanout;
    inline static constexpr auto s_subdivisions = box_t::s_subdivisions;

public:
    ndtree(
        std::span<sample_t>       collection,
        depth_t const             max_depth,
        size_type const           box_capacity,
        std::optional<boundary_t> limits = std::nullopt
    ) :
        m_data_view{ collection },
        m_box(
            limits.has_value() ? limits.value() : detail::compute_limits(collection),
            box_capacity,
            0uz,
            max_depth,
            nullptr
        ),
        m_max_depth{ max_depth },
        m_capacity{ box_capacity }
    {
        for (auto const& e : collection)
        {
            [[maybe_unused]]
            const auto inserted = insert(&e);
            assert(inserted);
        }
    }

    [[nodiscard]]
    inline auto insert(sample_t const* const sp) noexcept -> bool
    {
        return m_box.insert(sp);
    }

    inline auto reorganize() noexcept -> void
    {
        m_box.reorganize();
    }

    inline auto cache_summary() noexcept -> void
    {
        m_box.cache_summary();
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
        if (m_box.summary().has_value())
        {
            os << "Summary: " << m_box.summary().value().repr() << '\n';
        }
        os << "<\\ndtree<" << s_fanout << ", " << s_dimension << ">>\n";
    }

    [[nodiscard]]
    inline auto box() const -> box_t const&
    {
        return m_box;
    }

private:
    std::span<sample_t> m_data_view;
    box_t               m_box;
    depth_t             m_max_depth;
    size_type           m_capacity;
};

template <std::size_t Fanout, concepts::sample_concept Sample_Type>
auto operator<<(std::ostream& os, ndtree<Fanout, Sample_Type> const& tree)
    -> std::ostream&
{
    tree.print_info(os);
    tree.box().print_info(os);
    return os;
}

} // namespace ndt
