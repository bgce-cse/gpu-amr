#include "containers/static_vector.hpp"
#include "morton/morton_id.hpp"
#include "ndtree/ndtree.hpp"
#include "ndtree/ndhierarchy.hpp"
#include "ndtree/print_tree_a.hpp"
#include "utility/random.hpp"
#include <cstddef>
#include <iostream>
#include <tuple>
#include <iomanip>  // for std::setw, std::setprecision
#include <limits>   // for std::numeric_limits

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
    std::cout << "Hello balancing world\n";
    constexpr size_t N = 4;
    constexpr size_t M = 4;
    constexpr size_t Halo = 1;
    // using linear_index_t    = std::uint32_t;
    [[maybe_unused]]
    constexpr auto Fanout = 2;
    using tensor_t        = amr::containers::static_tensor<float, Halo,  N, M>;

    using patch_index_t       = amr::ndt::morton::morton_id<7u, 2u>;
    using tree_t        = amr::ndt::tree::ndtree<cell, patch_index_t, tensor_t>;

    tree_t h(100000000); // Provide initial capacity

    ndt::print::example_patch_print<Halo , M ,N> printer("debug_tree");
 
    auto refine_criterion = [](const patch_index_t& idx)
    {
        auto [coords, level] = patch_index_t::decode(idx.id());
        auto max_size        = 1u << idx.max_depth();
        auto cell_size       = 1u << (idx.max_depth() - level);

        double mid_x  = coords[0] + 0.5 * cell_size;
        double mid_y  = coords[1] + 0.5 * cell_size;
        double center = 0.5 * max_size;
        double dist2  = (mid_x - center) * (mid_x - center) +
                       (mid_y - center) * (mid_y - center);

        // Only refine if not at max level!
        if (idx.level() == 0 || (level < idx.max_depth() &&
                                 dist2 < 0.3 / idx.level() * max_size * max_size))
        {
            return tree_t::refine_status_t::Refine;
        }
        return tree_t::refine_status_t::Stable;
    };


    // auto coarsen_criterion =[](const patch_index_t& idx)
    //         {
    //             auto [coords, level] = patch_index_t::decode(idx.id());
    //             auto max_size        = 1u << idx.max_depth();
    //             auto cell_size       = 1u << (idx.max_depth() - level);

    //             double mid_x  = coords[0] + 0.5 * cell_size;
    //             double mid_y  = coords[1] + 0.5 * cell_size;
    //             double center = 0.5 * max_size;
    //             double dist2  = (mid_x - center) * (mid_x - center) +
    //                            (mid_y - center) * (mid_y - center);

    //             // Only coarsen if not at min level!
    //             if (level > 0 && dist2 < 0.3 / idx.level() * max_size * max_size)
    //             {
    //                 return tree_t::refine_status_t::Coarsen;
    //             }
    //             return tree_t::refine_status_t::Stable;
    //         };

    
    for(size_t idx = 0; idx < h.size(); idx++){
        // Access S1 values (float)
        tensor_t& s1_patch = h.template get_patch<S1>(idx);

        for(size_t linear_idx = 0; linear_idx < N*M; linear_idx++) {
            auto padded_index = tensor_t::layout_t::logical_to_full_index(linear_idx);
            s1_patch[padded_index] = static_cast<float>(linear_idx);
        }

    }
    for(size_t idx = 0; idx < h.size(); idx++){
        // Access S1 values (float)
        std::cout << "patch " << idx << std::endl;
        tensor_t& s1_patch = h.template get_patch<S1>(idx);

        for(size_t linear_idx = 0; linear_idx < 36; linear_idx++) {

            std::cout << s1_patch[linear_idx] << std::endl; 
        }

    }
// 
    printer.print(h, "_iteration_0.vtk");

    int i = 1;
    for (; i != 6; ++i)
    {
        h.reconstruct_tree(refine_criterion);
        std::string file_extension = "_iteration_" + std::to_string(i) + ".vtk";
        printer.print(h, file_extension);
    }

    // for(size_t idx = 0; idx < h.size(); idx++){
    //     // Access S1 values (float)
    //     std::cout << "patch " << idx << std::endl;
    //     tensor_t& s1_patch = h.template get_patch<S1>(idx);

    //     for(size_t linear_idx = 0; linear_idx < 36; linear_idx++) {

    //         std::cout << s1_patch[linear_idx] << std::endl; 
    //     }

    // }


    // for (; i != 2; ++i)
    // {
    //     h.reconstruct_tree(coarsen_criterion);
    //     std::string file_extension = "_iteration_" + std::to_string(i) + ".vtk";
    //     printer.print(h, file_extension);
    // }



    std::cout << "adios balancing world\n";
    return EXIT_SUCCESS;
}
