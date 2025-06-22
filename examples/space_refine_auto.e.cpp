#include "containers/static_vector.hpp"
#include "ndtree/ndhierarchy.hpp"
#include "ndtree/ndtree.hpp"
#include "ndtree/structured_print.hpp"
#include "utility/random.hpp"
#include "morton/morton_id.hpp"
#include <cstddef>
#include <iostream>

template <std::floating_point F, std::uint8_t N>
struct cell
{
    amr::containers::static_vector<F, N> v;
};

template <std::floating_point F, std::uint8_t N>
inline auto operator<<(std::ostream& os, cell<F, N> const& c) -> std::ostream&
{
    return os << c.v;
}

int main()
{
    std::cout << "Hello autoamic space partition world\n";

    constexpr auto N = 2u;
    using F          = float;

    using rng = typename utility::random::srandom;
    rng::seed<unsigned long>();

    using rngf = typename utility::random::srandom;
    rngf::seed<F>();

    using cell_t  = cell<F, N>;
    using index_t = amr::ndt::morton::morton_id<10u, 2u>;
    using tree_t  = amr::ndt::tree::ndtree<cell_t, index_t>;
    tree_t h;

    ndt::print::vtk_print vtk_printer("output_auto_partitioning");
    ndt::print::structured_print structured_printer(std::cout);


    int i = 0;
    std::string file_extension = std::to_string(i) + ".vtk";
    vtk_printer.print(h,file_extension);
    for (; i != 8; ++i)
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

            // Only refine if not at max level!
            if (level < idx.max_depth() && dist2 < 0.3 / idx.level() * max_size*  max_size ) {
                return 1;
            }
            return 0;
            });
        h.apply_refine_coarsen();
        
        file_extension = std::to_string(i+1) + ".vtk";
        vtk_printer.print(h,file_extension);
        structured_printer.print(h);
    }
    for (; i != 16; ++i)
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

            // Only refine if not at max level!
            if (level < idx.max_depth() && dist2 < 0.3 / idx.level() * max_size*  max_size ) {
                return 2;
            }
            return 0;
            });
        h.apply_refine_coarsen();
        
        file_extension = std::to_string(i+1) + ".vtk";
        vtk_printer.print(h,file_extension);
        structured_printer.print(h);
    }
    
    
    return EXIT_SUCCESS;
}
