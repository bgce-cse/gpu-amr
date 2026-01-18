#ifndef AMR_INCLUDED_NDT_VTK_PRINT
#define AMR_INCLUDED_NDT_VTK_PRINT

#include "ndtree/patch_utils.hpp"
#include "utility/constexpr_functions.hpp"
#include "utility/error_handling.hpp"
#include "utility/logging.hpp"
#include "utility/utility_concepts.hpp"
#include <filesystem>
#include <fstream>

namespace amr::ndt::print
{

template <typename Physics_System>
struct vtk_print
{
public:
    using physics_system_t = Physics_System;

    explicit vtk_print(std::string base_filename)
        : m_base_filename{ std::move(base_filename) }
    {
        DEFAULT_SOURCE_LOG_TRACE("Initializing vtk output dir");
        std::filesystem::create_directory("vtk_output");
    }

    auto print(auto const& tree, std::string filename_extension) const -> void
    {
        std::string full_filename = "vtk_output/" + m_base_filename + filename_extension;
        std::ofstream file(full_filename);
        if (!file.is_open())
        {
            DEFAULT_SOURCE_LOG_ERROR("Could not open vtk output directory");
            throw std::runtime_error("Cannot open file: " + full_filename);
        }
        write_header(file);
        write_points(file, tree);
    }

private:
    auto write_header(std::ofstream& file) const -> void
    {
        file << "# vtk DataFile Version 3.0\n";
        file << "AMR Tree Structure\n";
        file << "ASCII\n";
        file << "DATASET UNSTRUCTURED_GRID\n";
    }

    auto write_points(std::ofstream& file, auto const& tree) const -> void
    {
        using tree_t         = std::remove_cvref_t<decltype(tree)>;
        using index_t        = typename tree_t::linear_index_t;
        using patch_layout_t = typename tree_t::patch_layout_t;
        using data_layout_t  = typename patch_layout_t::data_layout_t;
        using value_type =
            std::tuple_element<0, typename tree_t::deconstructed_raw_map_types_t>;
        using lc_interior_t = patch_layout_t::interior_iteration_t;

        constexpr auto dim = std::remove_cvref_t<decltype(tree)>::rank();
        constexpr auto box_points =
            static_cast<index_t>(utility::cx_functions::pow(2, dim)); // Changed from pow(dim, 2)
        static_assert(dim == 2 || dim == 3);
        constexpr auto halo_width = patch_layout_t::halo_width();
        const auto     tree_size  = tree.size();
        const auto     cell_count = tree_size * data_layout_t::elements();

        static constexpr auto type_repr = []<utility::concepts::Arithmetic T>
        {
            if constexpr (std::is_same_v<T, float>)
            {
                return "float";
            }
            else if constexpr (std::is_same_v<T, double>)
            {
                return "double";
            }
            else if constexpr (std::is_same_v<T, int>)
            {
                return "int";
            }
            utility::error_handling::assert_unreachable();
        };

        file << "POINTS " << cell_count * box_points << ' '
             << type_repr.template operator()<typename value_type::type::type>() << '\n';

        if constexpr (dim == 2)
        {
            for (std::size_t i = 0; i != tree_size; ++i)
            {
                const auto patch_id     = tree.get_node_index_at(i);
                const auto patch_origin = physics_system_t::patch_coord(patch_id);
                const auto cell_size    = physics_system_t::cell_sizes(patch_id);

                amr::containers::manipulators::shaped_for<lc_interior_t>(
                    [&file, &cell_size, &patch_origin, halo_width](auto const& idxs)
                    {
                        const auto cell_x =
                            patch_origin[0] +
                            static_cast<double>(idxs[1] - halo_width) * cell_size[0];
                        const auto cell_y =
                            patch_origin[1] +
                            static_cast<double>(idxs[0] - halo_width) * cell_size[1];

                        file << cell_x << ' ' << cell_y << ' ' << 0.0 << '\n';
                        file << cell_x + cell_size[0] << ' ' << cell_y << ' ' << 0 << '\n';
                        file << cell_x + cell_size[0] << ' ' << cell_y + cell_size[1] << ' '
                             << 0.0 << '\n';
                        file << cell_x << ' ' << cell_y + cell_size[1] << ' ' << 0 << '\n';
                    }
                );
            }
        }
        else // dim == 3
        {
               
            for (std::size_t i = 0; i != tree_size; ++i)
            {
              
                const auto patch_id     = tree.get_node_index_at(i);
                const auto patch_origin = physics_system_t::patch_coord(patch_id);
                const auto cell_size    = physics_system_t::cell_sizes(patch_id);
                
                amr::containers::manipulators::shaped_for<lc_interior_t>(
                    [&](auto const& idxs)
                    {
                        const auto cell_x =
                            patch_origin[0] +
                            static_cast<double>(idxs[2] - halo_width) * cell_size[0];
                        const auto cell_y =
                            patch_origin[1] +
                            static_cast<double>(idxs[1] - halo_width) * cell_size[1];
                        const auto cell_z =
                            patch_origin[2] +
                            static_cast<double>(idxs[0] - halo_width) * cell_size[2];

                        // 8 corners of a hexahedron (VTK_VOXEL ordering)
                        file << cell_x << ' ' << cell_y << ' ' << cell_z << '\n';
                        file << cell_x + cell_size[0] << ' ' << cell_y << ' ' << cell_z << '\n';
                        file << cell_x << ' ' << cell_y + cell_size[1] << ' ' << cell_z << '\n';
                        file << cell_x + cell_size[0] << ' ' << cell_y + cell_size[1] << ' ' << cell_z << '\n';
                        file << cell_x << ' ' << cell_y << ' ' << cell_z + cell_size[2] << '\n';
                        file << cell_x + cell_size[0] << ' ' << cell_y << ' ' << cell_z + cell_size[2] << '\n';
                        file << cell_x << ' ' << cell_y + cell_size[1] << ' ' << cell_z + cell_size[2] << '\n';
                        file << cell_x + cell_size[0] << ' ' << cell_y + cell_size[1] << ' ' << cell_z + cell_size[2] << '\n';
                    }
                );
            }
        }

        file << "CELLS " << cell_count << ' ' << cell_count * (box_points + 1) << '\n';
        for (std::size_t i = 0; i != cell_count; ++i)
        {
            file << box_points;
            for (auto j = index_t{}; j != box_points; ++j)
            {
                file << ' ' << i * box_points + j;
            }
            file << '\n';
        }

        file << "CELL_TYPES " << cell_count << '\n';
        for (std::size_t i = 0; i != cell_count; ++i)
        {
            if constexpr (dim == 2)
            {
                file << "9\n";  // VTK_QUAD
            }
            else // dim == 3
            {
                file << "11\n"; // VTK_VOXEL
            }
        }

        file << "CELL_DATA " << cell_count << '\n';
        file << "SCALARS cell_index int 1\n";
        file << "LOOKUP_TABLE default\n";
        for (std::size_t i = 0; i != cell_count; ++i)
        {
            file << i << '\n';
        }

        file << "SCALARS is_halo int 1\n";
        file << "LOOKUP_TABLE default\n";
        for (std::size_t i = 0; i != tree_size; ++i)
        {
            for (auto j = typename data_layout_t::size_type{};
                 j != data_layout_t::elements();
                 ++j)
            {
                file << utils::patches::is_halo_cell<patch_layout_t>(j) << '\n';
            }
        }

        [&tree, &file, tree_size, cell_count]<std::size_t... I>(std::index_sequence<I...>)
        {
            (((void)I,
              [&tree, &file, tree_size, cell_count]()
              {
                  using element_type = typename std::tuple_element<
                      I,
                      typename tree_t::deconstructed_raw_map_types_t>::type;
                  file << "SCALARS " << element_type::name() << ' '
                       << type_repr.template operator()<typename element_type::type>()
                       << " 1\n";
                  file << "LOOKUP_TABLE default\n";
                  for (size_t i = 0; i != tree_size; ++i)
                  {
                      auto const& patch = tree.template get_patch<element_type>(i);
                      amr::containers::manipulators::for_each<lc_interior_t>(
                          patch.data(),
                          [&file](auto const& p, auto const& idxs)
                          { file << p[idxs] << '\n'; }
                      );
                  }
              }()),
             ...);
        }(std::make_index_sequence<
            std::tuple_size_v<typename tree_t::deconstructed_raw_map_types_t>>{});
    }

    std::string m_base_filename;
};

} // namespace amr::ndt::print

#endif // AMR_INCLUDED_NDT_VTK_PRINT
