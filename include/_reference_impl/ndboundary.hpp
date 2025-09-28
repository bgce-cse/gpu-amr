#ifndef GPU_AMR_NDBOUNDARY
#define GPU_AMR_NDBOUNDARY

#include "ndt_concepts.hpp"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <ranges>

namespace ndt
{

template <ndt_concepts::point Point_Type>
class NDBoundary
{
public:
    using point_t                            = Point_Type;
    using value_type                         = typename point_t::value_type;
    inline static constexpr auto s_dimension = point_t::s_dimension;
    using index_t                            = std::remove_const_t<decltype(s_dimension)>;

public:
    constexpr NDBoundary() noexcept = delete;

    constexpr NDBoundary(point_t const& p1, point_t const& p2) noexcept
    {
        for (index_t i = 0; i != s_dimension; ++i)
        {
            // Explicit copy to avoid dangling reference to a temporary
            const std::pair<value_type, value_type> r = std::minmax(p1[i], p2[i]);
            m_min[i]                                  = r.first;
            m_max[i]                                  = r.second;
        }
    }

    constexpr NDBoundary(value_type v1, value_type v2) noexcept
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
    constexpr auto operator<=>(NDBoundary const&) const = default;

private:
    point_t m_min;
    point_t m_max;
};

template <ndt_concepts::point Point_Type>
auto operator<<(std::ostream& os, NDBoundary<Point_Type> const& b) -> std::ostream&
{
    os << "{ " << b.min() << " }, { " << b.max() << " }";
    return os;
}

namespace detail
{

template <ndt_concepts::point Point_Type>
[[nodiscard]]
constexpr auto
    in(Point_Type const&               p,
       NDBoundary<Point_Type> const&   b,
       typename Point_Type::value_type tol) noexcept -> bool
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

template <ndt_concepts::point Point_Type>
[[nodiscard]]
constexpr auto count_in(
    std::ranges::range auto const& collection,
    NDBoundary<Point_Type> const&  b
) noexcept -> std::size_t
{
    return std::ranges::count_if(collection, [b](auto const& p) { return in(p, b); });
}

[[nodiscard]]
auto compute_limits(std::ranges::range auto const& data) noexcept
    requires ndt_concepts::cell<std::ranges::range_value_t<decltype(data)>>
{
    using sample_t   = std::ranges::range_value_t<decltype(data)>;
    using point_t    = typename sample_t::position_t;
    constexpr auto N = point_t::s_dimension;
    point_t        min;
    point_t        max;
    for (auto i = decltype(N){ 0 }; i != N; ++i)
    {
        const auto bounds = std::ranges::minmax(
            data |
            std::views::transform([i](auto const& p) -> auto { return p.position()[i]; })
        );
        min[i] = bounds.min;
        max[i] = bounds.max;
    }
    return NDBoundary<point_t>{ min, max };
}

} // namespace detail

} // namespace ndt

#endif // GPU_AMR_NDBOUNDARY
