#ifndef AMR_INCLUDED_NDT_STRUCTURED_PRINT
#define AMR_INCLUDED_NDT_STRUCTURED_PRINT

#include "ndtree.hpp"
#include <algorithm>
#include <bitset>
#include <ostream>
#include <ranges>

namespace ndt::print
{

struct structured_print
{
public:
    structured_print(std::ostream& os) noexcept
        : m_os(os)
    {
    }

    auto print(auto const& tree) const -> void
    {
        using tree_t    = std::remove_cvref_t<decltype(tree)>;
        std::vector cpy = auto(tree.blocks());
        std::ranges::sort(
            cpy, [](auto const& a, auto const& b) { return a.operator<(b); }
        );
        for ([[maybe_unused]] auto const& [h, p, md] : cpy)
        {
            using index_t = decltype(h);
            print_header(m_os, index_t::level(h))
                << "h: " << std::bitset<index_t::bits()>(h.id()).to_string()
                << ", offset: " << decltype(h)::offset_of(h) << ", ptr: " << p << '\n';
            for (auto i = decltype(tree_t::s_nd_fanout){}; i != tree_t::s_nd_fanout; ++i)
            {
                print_header(m_os, index_t::level(h))
                    << "@" << i << ": " << p[i] << " ("
                    << (md[i].alive ? "alive" : "dead") << ")" << '\n';
            }
        }
    }

private:
    static auto print_header(std::ostream& os, auto depth) -> std::ostream&
    {
        for (auto i = decltype(depth){}; i != depth; ++i)
        {
            os << '\t';
        }
        return os;
    }

    std::ostream& m_os;
};

} // namespace ndt::print

#endif // AMR_INCLUDED_NDT_STRUCTURED_PRINT
