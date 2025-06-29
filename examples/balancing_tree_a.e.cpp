#include "ndtree/flat_ndtree.hpp"
#include "containers/static_vector.hpp"
#include "morton/morton_id.hpp"
#include "ndtree/ndhierarchy.hpp"
#include "ndtree/print_tree_a.hpp"
#include "utility/random.hpp"
#include <cstddef>
#include <iostream>
#include <tuple>

// --- Cell type from flat_ndtree.e.cpp ---
struct S1
{
    using type = float;
    static constexpr auto index() noexcept -> std::size_t { return 0; }
    type value;
};

struct S2
{
    using type = int;
    static constexpr auto index() noexcept -> std::size_t { return 1; }
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

    auto data_tuple() -> auto& { return m_data; }
    auto data_tuple() const -> auto const& { return m_data; }

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

    using index_t = amr::ndt::morton::morton_id<8u, 2u>;
    using tree_t  = amr::ndt::tree::flat_ndtree<cell, index_t>;
    tree_t h(100000000); // Provide initial capacity

    ndt::print::vtk_print vtk_printer("balancing_tree_a");

    int i = 0;

    for (; i != 7; ++i)
    {
        std::cout << "starting with i : "<< i <<std::endl;
        h.compute_refine_flag([](const index_t& idx){
            auto [coords, level] = index_t::decode(idx.id());
            auto max_size = 1u << idx.max_depth();
            auto cell_size = 1u << (idx.max_depth() - level);

            double mid_x = coords[0] + 0.5 * cell_size;
            double mid_y = coords[1] + 0.5 * cell_size;
            double center = 0.5 * max_size;
            double dist2 = (mid_x - center) * (mid_x - center) +
                           (mid_y - center) * (mid_y - center);

            // Only refine if not at max level!
            if (level < idx.max_depth() && dist2 < 0.3 / idx.level() * max_size*  max_size ) {
                return 1;
            }
            return 0;
        });
        auto [to_refine, to_coarsen] = h.apply_refine_coarsen();

        std::cout << "apply refine coarseing done " << std::endl;
        h.balancing(to_refine, to_coarsen);
        std::cout << "balancing done" <<std::endl;
        h.fragment(to_refine);
        h.recombine(to_coarsen);

        std::string file_extension = std::to_string(i) + ".vtk";
        vtk_printer.print(h, file_extension);
    }
    for (; i != 14; ++i)
    {
        h.compute_refine_flag([](const index_t& idx){
            auto [coords, level] = index_t::decode(idx.id());
            auto max_size = 1u << idx.max_depth();
            auto cell_size = 1u << (idx.max_depth() - level);

            double mid_x = coords[0] + 0.5 * cell_size;
            double mid_y = coords[1] + 0.5 * cell_size;
            double center = 0.5 * max_size;
            double dist2 = (mid_x - center) * (mid_x - center) +
                           (mid_y - center) * (mid_y - center);

            // Only coarsen if not at min level!
            if (level > 0 && dist2 < 0.3 / idx.level() * max_size*  max_size ) {
                return 2;
            }
            return 0;
        });
        std::cout << "compute flags done done" <<std::endl;
        auto [to_refine, to_coarsen] = h.apply_refine_coarsen();
        std::cout << "apply refine coarseing done " << std::endl;
        std::cout << "size of coarsen vector beofre balancing " << to_coarsen.size() <<std::endl;
        h.balancing(to_refine, to_coarsen);
        std::cout << "size of coarsen vector after balancing " << to_coarsen.size() <<std::endl;
        std::cout << "balancing done" <<std::endl;
        h.fragment(to_refine);
        h.recombine(to_coarsen);
        std::cout << "actual refine / coarsening done" <<std::endl;
        std::string file_extension = std::to_string(i) + ".vtk";
        vtk_printer.print(h, file_extension);
    }

    return EXIT_SUCCESS;
}
