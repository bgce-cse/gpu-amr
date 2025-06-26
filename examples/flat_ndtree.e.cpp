#include "ndtree/flat_ndtree.hpp"
#include "containers/static_vector.hpp"
#include "morton/morton_id.hpp"
#include "ndtree/ndhierarchy.hpp"
#include "ndtree/structured_print.hpp"
#include "utility/random.hpp"
#include <cstddef>
#include <iostream>
#include <tuple>

struct S1
{
    using type = float;

    static constexpr auto index() noexcept -> std::size_t
    {
        return 0;
    }
};

struct S2
{
    using type = int;

    static constexpr auto index() noexcept -> std::size_t
    {
        return 1;
    }
};

struct cell
{
    using deconstructed_types_map_t = std::tuple<S1, S2>;
};

int main()
{
    std::cout << "Hello space partition world\n";

    using rngi = typename utility::random::srandom;
    rngi::seed<typename S2::type>();

    using rngf = typename utility::random::srandom;
    rngf::seed<typename S1::type>();

    using index_t = amr::ndt::morton::morton_id<7u, 2u>;
    using tree_t  = amr::ndt::tree::flat_ndtree<cell, index_t>;
    tree_t h(10);

    std::cout << h.get<S1>() << '\n';
    std::cout << h.get<S2>() << '\n';

    return EXIT_SUCCESS;
}
