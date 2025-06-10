#include "containers/static_vector.hpp"
#include "ndtree/ndhierarchy.hpp"
#include "ndtree/ndtree.hpp"
#include "ndtree/structured_print.hpp"
#include "utility/random.hpp"
#include <iostream>

template <std::floating_point F, std::uint8_t N>
struct cell
{
    amr::containers::static_vector<N, F> v;
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
    rng::seed<F>();

    using cell_t = cell<F, N>;
    using tree_t = amr::ndt::tree::ndtree<cell_t, N, 2u, 7u>;
    tree_t h;

    ndt::print::structured_print printer(std::cout);

    for (int i = 0; auto const& m : tree_t::hierarchy_id_t::s_generation_masks)
    {
        std::cout << "i: " << i++ << ' ' << std::bitset<7 * 2>(m).to_string() << '\n';
    }
    for (int i = 0; auto const& m : tree_t::hierarchy_id_t::s_predecessor_masks)
    {
        std::cout << "i: " << i++ << ' ' << std::bitset<7 * 2>(m).to_string() << '\n';
    }

    for (auto i = 0; i != 10; ++i)
    {
        auto [id, ptr] = h.fragment(
            h.blocks()[rng::randrange(0uz, h.blocks().size() - 1)].id,
            rng::randrange(0u, 3u)
        );
        for (auto j = 0; j != tree_t::s_nd_fanout; ++j)
        {
            auto _ = new (&ptr[j]) cell_t{
                .v = { rngf::randnormal(F(0), F(1)), rngf::randnormal(F(0), F(1)) }
            };
        }
    }

    printer.print(h);

    return EXIT_SUCCESS;
}
