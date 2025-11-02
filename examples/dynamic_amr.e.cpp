#include "containers/container_utils.hpp"
#include "containers/static_layout.hpp"
#include "containers/static_shape.hpp"
#include "containers/static_vector.hpp"
#include "morton/morton_id.hpp"
#include "ndtree/ndhierarchy.hpp"
#include "ndtree/ndtree.hpp"
#include "ndtree/patch_layout.hpp"
#include "ndtree/print_tree_a.hpp"
#include "utility/random.hpp"
#include <cstddef>
#include <iomanip> // for std::setw, std::setprecision
#include <iostream>
#include <limits> // for std::numeric_limits
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

int main()
{
    std::cout << "Hello dynamic amr world\n";
    constexpr std::size_t N    = 2;
    constexpr std::size_t M    = 6;
    constexpr std::size_t Halo = 1;
    // using linear_index_t    = std::uint32_t;
    [[maybe_unused]]
    constexpr auto Fanout = 2;
    using shape_t         = amr::containers::static_shape<N, M>;
    using layout_t        = amr::containers::static_layout<shape_t>;
    using index_t         = typename layout_t::index_t;

    using patch_index_t  = amr::ndt::morton::morton_id<10u, 2u>;
    using patch_layout_t = amr::ndt::patches::patch_layout<layout_t, Halo>;
    using tree_t         = amr::ndt::tree::ndtree<cell, patch_index_t, patch_layout_t>;

    tree_t tree(1000000); // Provide initial capacity

    ndt::print::example_patch_print<Halo, M, N> printer("dynamic_amr_tree");

    auto amr_condition = [](const patch_index_t& idx, int step, int max_step)
    {
        auto [coords, level] = patch_index_t::decode(idx.id());
        auto max_size        = 1u << idx.max_depth();
        auto patch_size      = 1u << (idx.max_depth() - level);

        double start_x = coords[0]; // Convert to absolute coords
        double end_x   = start_x + patch_size;

        double step_x = max_size * (static_cast<double>(step) / max_step);

        bool in_interval_1 = step_x < end_x + patch_size && step_x > start_x - patch_size;
        bool in_interval_2 =
            step_x < end_x + 2 * patch_size && step_x > start_x - 2 * patch_size;

        bool refine  = in_interval_1 && level < idx.max_depth();
        bool coarsen = !in_interval_2 && level > 0;

        if (refine)
        {
            return tree_t::refine_status_t::Refine;
        }

        if (coarsen)
        {
            return tree_t::refine_status_t::Coarsen;
        }

        return tree_t::refine_status_t::Stable;
    };

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
    printer.print(tree, "_iteration_0.vtk");

    int max_steps = 100;

    int i = 1;
    for (; i != max_steps; ++i)
    {
        auto amr_condition_with_time =
            [&amr_condition, &i, max_steps](const patch_index_t& idx)
        {
            return amr_condition(idx, i, max_steps);
        };
        tree.reconstruct_tree(amr_condition_with_time);
        std::string file_extension = "_iteration_" + std::to_string(i) + ".vtk";
        printer.print(tree, file_extension);
    }

    std::cout << "Adiós dynamic amr world\n";
    return EXIT_SUCCESS;
}
