#include "containers/static_layout.hpp"
#include "containers/static_shape.hpp"
#include "morton/morton_id.hpp"
#include "ndtree/ndtree.hpp"
#include "ndtree/patch_layout.hpp"
#include "ndtree/structured_print.hpp"
#include "utility/random.hpp"
#include <cstddef>
#include <iostream>
#include <string_view>
#include <tuple>

struct S1
{
    using type = float;

    static constexpr auto index() noexcept -> std::size_t
    {
        return 0;
    }

    static constexpr auto name() noexcept -> std::string_view
    {
        return "S1";
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

    static constexpr auto name() noexcept -> std::string_view
    {
        return "S2";
    }

    type value;
};

struct cell
{
    using deconstructed_types_map_t = std::tuple<S1, S2>;

    cell(typename S1::type v1 = 0, typename S2::type v2 = 0)
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

// --- End cell type ---

int main()
{
    constexpr std::size_t N    = 4;
    constexpr std::size_t M    = 6;
    constexpr std::size_t Halo = 2;
    // using linear_index_t    = std::uint32_t;
    [[maybe_unused]]
    constexpr auto Fanout = 2;
    using shape_t         = amr::containers::static_shape<N, M>;
    using layout_t        = amr::containers::static_layout<shape_t>;
    using index_t         = typename layout_t::index_t;

    using patch_index_t  = amr::ndt::morton::morton_id<9u, 2u>;
    using patch_layout_t = amr::ndt::patches::patch_layout<layout_t, Halo>;
    using tree_t         = amr::ndt::tree::ndtree<cell, patch_index_t, patch_layout_t>;

    tree_t tree(100000);

    amr::ndt::print::structured_print p(std::cout);
    amr::ndt::print::vtk_print        vtk_printer("test");

    p.print(tree);

    auto refine_criterion = [](const patch_index_t& idx)
    {
        const auto prob = 0.3f;
        const auto r    = utility::random::srandom::randfloat<float>();
        if (idx.level() == 0 || (r < prob))
        {
            return tree_t::refine_status_t::Refine;
        }
        else
        {
            return tree_t::refine_status_t::Stable;
        }
    };

    std::cout << "patch size: " << patch_layout_t::flat_size() << '\n';

    int ii = 0;
    for (std::size_t idx = 0; idx < tree.size(); idx++)
    {
        // Access S1 values (float)
        auto& s1_patch = tree.template get_patch<S1>(idx);

        for (index_t linear_idx = 0; linear_idx != patch_layout_t::flat_size();
             ++linear_idx)
        {
            if (amr::ndt::utils::patches::is_halo_cell<patch_layout_t>(linear_idx))
            {
                continue;
            }
            s1_patch[linear_idx] = static_cast<float>(ii++);
        }
    }

    std::cout << "Print 1\n";
    p.print(tree);

    tree.halo_exchange_update();

    std::cout << "Print 2\n";
    p.print(tree);

    for (int i = 0; i != 2; ++i)
    {
        tree.reconstruct_tree(refine_criterion);
        tree.halo_exchange_update();
    }

    std::cout << "Print 3\n";
    p.print(tree);

    std::string file_extension = "_iteration_" + std::to_string(0) + ".vtk";
    vtk_printer.print(tree, file_extension);

    return EXIT_SUCCESS;
}
