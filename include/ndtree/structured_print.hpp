#ifndef AMR_INCLUDED_NDT_STRUCTURED_PRINT
#define AMR_INCLUDED_NDT_STRUCTURED_PRINT

#include "ndtree.hpp"
#include <algorithm>
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
        std::vector cpy = auto(tree.blocks());
        std::ranges::sort(
            cpy, [](auto const& a, auto const& b) { return a.operator<(b); }
        );
        for (auto const& [h, p] : cpy)
        {
            print_header(m_os, h.generation())
                << "h: " << h.id().to_string() << ", gen id: " << h.generation_id()
                << ", ptr: " << p << '\n';
            for (int i = 0; i != std::remove_cvref_t<decltype(tree)>::s_nd_fanout; ++i)
            {
                print_header(m_os, h.generation()) << "@" << i << ": " << p[i] << '\n';
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
