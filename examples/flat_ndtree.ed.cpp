#include "containers/static_vector.hpp"
#include "morton/morton_id.hpp"
#include "ndtree/ndhierarchy.hpp"
#include "ndtree/ndtree.hpp"
#include "ndtree/patch.hpp"
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
    ndt::print::vtk_print vtk_printer("tree_a_test");
    using rngi = typename utility::random::srandom;
    rngi::seed<typename S2::type>();

    using rngf = typename utility::random::srandom;
    rngf::seed<typename S1::type>();

    using Patch_Type    = PatchIndexer<4, 4>;
    using index_t       = amr::ndt::morton::morton_id<7u, 2u>;
    using tree_t        = amr::ndt::tree::ndtree<cell, index_t, Patch_Type>;
    const auto capacity = 100;
    tree_t     tree(capacity);

    tree.get<S1>(0) = 10;
    tree.get<S2>(0) = 100;

    for (auto i = 0uz; i != tree.size(); ++i)
    {
        std::cout << tree.get<S1>(i) << '\n';
        std::cout << tree.get<S2>(i) << '\n';
    }

    std::cout << tree.size() << '\n';

    for (int j = 0; j != 10; ++j)
    {
        auto chosen = rngi::randrange(0uz, tree.size() - 1);
        std::cout << "Chose: " << chosen << '\n';
        tree.fragment(tree.get_node_index_at(chosen));
        std::cout << "Pre compact: \n";
        for (auto i = 0uz; i != tree.size(); ++i)
        {
            std::cout << tree.get_node_index_at(i).repr() << '\n';
        }
        tree.compact();
        std::cout << "Post compact: \n";
        for (auto i = 0uz; i != tree.size(); ++i)
        {
            std::cout << tree.get_node_index_at(i).repr() << '\n';
        }
    }

    std::cout << "\nSorting...\n";
    for (auto i = 0uz; i != tree.size(); ++i)
    {
        std::cout << tree.get_node_index_at(i).repr() << '\n';
    }
    tree.sort_buffers();

    std::cout << "\nSorted...\n";
    for (auto i = 0uz; i != tree.size(); ++i)
    {
        std::cout << tree.get<S1>(i) << ", ";
        std::cout << tree.get<S2>(i) << '\n';
    }
    std::string file_extension = "refine.vtk";
    vtk_printer.print(tree, file_extension);

    file_extension = "coarsen.vtk";
    vtk_printer.print(tree, file_extension);
    return EXIT_SUCCESS;
}
