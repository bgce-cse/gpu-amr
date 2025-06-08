#include "containers/static_vector.hpp"
#include "ndtree/ndhierarchy.hpp"
#include "ndtree/structured_print.hpp"
#include "utility/random.hpp"
#include <iostream>

template <std::floating_point F, std::uint8_t N>
struct cell
{
    amr::containers::static_vector<N, F> v;
};

int main()
{
    std::cout << "Hello space partition world\n";

    constexpr auto N = 2u;
    using F          = float;

    using rng = typename utility::random::srandom::srandom;
    rng::seed<unsigned long>();

    using hierarchy_t = amr::ndt::hierarchy::ndhierarchy<cell<F, N>, N, 2u, 7u>;
    hierarchy_t h;

    ndt::print::structured_print printer(std::cout);

    for (int i = 0; auto const& m : hierarchy_t::hierarchy_id_t::s_generation_masks)
    {
        std::cout << "i: " << i++ << ' ' << std::bitset<7 * 2>(m).to_string() << '\n';
    }

    auto _ = h.fragment(h.members()[0].id, 1);
    for (auto i = 0; i != 10; ++i)
    {
        auto _ = h.fragment(
            h.members()[rng::randrange(0uz, h.members().size() - 1)].id,
            rng::randrange(0u, 3u)
        );
    }

    printer.print(h);

    return EXIT_SUCCESS;
}
