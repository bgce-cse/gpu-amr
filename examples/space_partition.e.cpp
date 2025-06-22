#include "containers/static_vector.hpp"
#include "ndtree/ndhierarchy.hpp"
#include "ndtree/ndtree.hpp"
#include "ndtree/structured_print.hpp"
#include "utility/random.hpp"
#include <cstddef>
#include <iostream>

template <std::floating_point F, std::uint8_t N>
struct cell
{
    amr::containers::static_vector<F, N> v;
};

template <std::floating_point F, std::uint8_t N>
inline auto operator<<(std::ostream& os, cell<F, N> const& c) -> std::ostream&
{
    return os << c.v;
}

int main()
{
    std::cout << "Hello space partition world\n";

    constexpr auto N = 2u;
    using F          = float;

    using rng = typename utility::random::srandom;
    rng::seed<unsigned long>();

    using rngf = typename utility::random::srandom;
    rngf::seed<F>();

    using cell_t  = cell<F, N>;
    using index_t = amr::ndt::hierarchy::hierarchical_prefix_id<N, 2u, 7u>;
    using tree_t  = amr::ndt::tree::ndtree<cell_t, index_t>;
    tree_t h;

    ndt::print::structured_print printer(std::cout);

    std::cout << "d: x "
              << std::bitset<7 * 2 + 3>(tree_t::node_index_t::s_depth_mask).to_string()
              << '\n';
    for (int i = 0; auto const& m : tree_t::node_index_t::s_generation_masks)
    {
        std::cout << "g: " << i++ << ' ' << std::bitset<7 * 2 + 3>(m).to_string() << '\n';
    }
    for (int i = 0; auto const& m : tree_t::node_index_t::s_predecessor_masks)
    {
        std::cout << "p: " << i++ << ' ' << std::bitset<7 * 2 + 3>(m).to_string() << '\n';
    }

    typename index_t::offset_t offset = rng::randrange(0u, 3u);
    auto                       bp     = h.blocks()[0];
    for (auto j = 0; j != index_t::nd_fanout(); ++j)
    {
        [[maybe_unused]]
        auto _ = new (&bp.ptr[j]) cell_t{
            .v = { rngf::randnormal(F(0), F(1)), rngf::randnormal(F(0), F(1)) }
        };
    }
    for (auto i = 0; i != 10; ++i)
    {
        auto fragment_id = index_t::child_of(bp.id, offset);
        bp               = h.fragment(fragment_id);

        for (auto j = 0; j != index_t::nd_fanout(); ++j)
        {
            [[maybe_unused]]
            auto _ = new (&bp.ptr[j]) cell_t{
                .v = { rngf::randnormal(F(0), F(1)), rngf::randnormal(F(0), F(1)) }
            };
        }

        while (true)
        {
            bp     = h.blocks()[rng::randrange(0uz, h.blocks().size() - 1)];
            offset = rng::randrange(0u, 3u);
            if (bp[offset].alive)
            {
                break;
            }
        }
    }

    printer.print(h);

    return EXIT_SUCCESS;
}
