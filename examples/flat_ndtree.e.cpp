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

    type value;
};

struct S2
{
    using type = int;

    static constexpr auto index() noexcept -> std::size_t
    {
        return 1;
    }

    type value;
};

struct cell
{
    using deconstructed_types_map_t = std::tuple<S1, S2>;

    cell(typename S1::type v1, typename S2::type v2)
    {
        std::get<S1>(m_data).value = v1;
        std::get<S2>(m_data).value = v2;
    }

    auto data_tuple() -> auto&
    {
        return m_data;
    }

    auto data_tuple() const -> auto const&
    {
        return m_data;
    }

    deconstructed_types_map_t m_data;
};

auto operator<<(std::ostream& os, cell const& c) -> std::ostream&
{
    return os << "S1: " << std::get<S1>(c.data_tuple()).value
              << ", S2: " << std::get<S2>(c.data_tuple()).value;
}

int main()
{
    std::cout << "Hello space partition world\n";

    using rngi = typename utility::random::srandom;
    rngi::seed<typename S2::type>();

    using rngf = typename utility::random::srandom;
    rngf::seed<typename S1::type>();

    using index_t   = amr::ndt::morton::morton_id<7u, 2u>;
    using tree_t    = amr::ndt::tree::flat_ndtree<cell, index_t>;
    const auto size = 10;
    tree_t     tree(size);

    for (auto i = 0; i != size; ++i)
    {
        std::cout << tree.get<S1>(i) << '\n';
        std::cout << tree.get<S2>(i) << '\n';
    }

    [[maybe_unused]]
    auto _ = tree.fragment(index_t::root());

    return EXIT_SUCCESS;
}
