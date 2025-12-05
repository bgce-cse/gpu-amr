#include "containers/static_layout.hpp"
#include "containers/static_shape.hpp"
#include "morton/morton_id.hpp"
#include "ndtree/ndtree.hpp"
#include "ndtree/patch_layout.hpp"
#include "ndtree/print_tree_a.hpp"
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

template <typename Tree, typename PatchLayout>
void increment_all_data_cells(Tree& tree)
{
    using index_t = typename PatchLayout::index_t;

    for (std::size_t idx = 0; idx < tree.size(); idx++)
    {
        auto& s1_patch = tree.template get_patch<S1>(idx);

        for (index_t linear_idx = 0; linear_idx != PatchLayout::flat_size(); ++linear_idx)
        {
            if (amr::ndt::utils::patches::is_halo_cell<PatchLayout>(linear_idx))
            {
                continue;
            }
            s1_patch[linear_idx] += 5.0f;
        }
    }
}

template <typename Tree, typename PatchLayout>
void initialize_data_cells(Tree& tree)
{
    using index_t = typename PatchLayout::index_t;

    int ii = 0;
    for (std::size_t idx = 0; idx < tree.size(); idx++)
    {
        auto& s1_patch = tree.template get_patch<S1>(idx);

        for (index_t linear_idx = 0; linear_idx != PatchLayout::flat_size(); ++linear_idx)
        {
            if (amr::ndt::utils::patches::is_halo_cell<PatchLayout>(linear_idx))
            {
                continue;
            }
            s1_patch[linear_idx] = static_cast<float>(ii++);
        }
    }
}

int main()
{
    constexpr std::size_t N    = 4;
    constexpr std::size_t M    = 8;
    constexpr std::size_t Halo = 1;

    using shape_t  = amr::containers::static_shape<N, M>;
    using layout_t = amr::containers::static_layout<shape_t>;
    // using index_t         = typename layout_t::index_t;
    using patch_index_t  = amr::ndt::morton::morton_id<9u, 2u>;
    using patch_layout_t = amr::ndt::patches::patch_layout<layout_t, Halo>;
    using tree_t         = amr::ndt::tree::ndtree<cell, patch_index_t, patch_layout_t>;

    tree_t tree(100000);

    amr::ndt::print::example_halo_patch_print<Halo, M, N> p1("tree_halo");
    amr::ndt::print::example_patch_print<Halo, M, N>      p2("tree_no_halo");

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

    std::cout << "Patch size: " << patch_layout_t::flat_size() << '\n';

    // Initialize data
    initialize_data_cells<tree_t, patch_layout_t>(tree);

    // Initial output
    std::cout << "Step 0: Initial state\n";
    p1.print(tree, "_step_0.vtk");
    p2.print(tree, "_step_0.vtk");

    // Main iteration loop
    constexpr int num_steps = 6;
    for (int step = 1; step <= num_steps; ++step)
    {
        std::cout << "Step " << step << ": ";

        // Refine on steps 1, 3, 5
        if (step % 2 == 1)
        {
            std::cout << "Refining and exchanging halos\n";
            tree.reconstruct_tree(refine_criterion);
            tree.halo_exchange_update();
        }
        else
        {
            std::cout << "Incrementing data and exchanging halos\n";
            increment_all_data_cells<tree_t, patch_layout_t>(tree);
            tree.halo_exchange_update();
        }

        // Output after this step
        std::string suffix = "_step_" + std::to_string(step) + ".vtk";
        p1.print(tree, suffix);
        p2.print(tree, suffix);
    }

    std::cout << "Simulation complete. Output files written to vtk_output/\n";

    return EXIT_SUCCESS;
}
