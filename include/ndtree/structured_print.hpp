#ifndef AMR_INCLUDED_NDT_STRUCTURED_PRINT
#define AMR_INCLUDED_NDT_STRUCTURED_PRINT

#include "containers/container_manipulations.hpp"
#include "ndtree/patch_utils.hpp"
#include "utility/constexpr_functions.hpp"
#include "utility/error_handling.hpp"
#include "utility/logging.hpp"
#include "utility/utility_concepts.hpp"
#include <algorithm>
#include <bitset>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <ostream>
#include <vector>

namespace amr::ndt::print
{

struct structured_print
{
public:
    structured_print(std::ostream& os) noexcept
        : m_os(os)
    {
    }

    auto print(auto const& tree) const -> void
    {
        using tree_t        = std::remove_cvref_t<decltype(tree)>;
        using patch_shape_t = typename tree_t::patch_layout_t::padded_layout_t::shape_t;
        using index_t       = typename patch_shape_t::index_t;
        using tree_index_t  = typename tree_t::patch_index_t;
        using map_t         = typename tree_t::deconstructed_raw_map_types_t;
        using lc_t          = amr::containers::control::
            loop_control<patch_shape_t, index_t{}, patch_shape_t::sizes(), index_t{ 1 }>;
        static constexpr auto rank = patch_shape_t::rank();

        static constexpr auto           size        = tree_index_t::max_depth();
        static constexpr decltype(size) k           = 2;
        static constexpr auto           prefix_pool = [] constexpr -> auto
        {
            std::array<char, size + k> arr{};
            std::ranges::fill(arr, '\t');
            arr[size]     = ' ';
            arr[size + 1] = '{';
            return arr;
        }();
        static constexpr auto prefixes = []
        {
            std::array<std::string_view, size> arr{};
            for (std::size_t d = 0; d != size; ++d)
            {
                arr[d] = std::string_view(prefix_pool.data() + size - d, d + k);
            }
            return arr;
        }();

        for (auto i = 0uz; i != tree.size(); ++i)
        {
            auto const h = tree.get_node_index_at(i);

            print_header(m_os, tree_index_t::level(h))
                << "h: " << std::bitset<tree_index_t::bits()>(h.id()).to_string()
                << ", offset: " << decltype(h)::offset_of(h) << '\n';

            containers::manipulators::shaped_for<lc_t>(
                [this, &tree, &h, i](auto const& idxs)
                {
                    auto prefix = [&h](auto r, auto const& iidxs) constexpr noexcept
                    {
                        // TODO: This was copied from tensor ostream
                        // Write a function that prints a tuple of tensors,
                        // which is what we need here
                        return (r == 0) ? (iidxs[rank - 1] == 0)
                                              ? prefixes[tree_index_t::level(h)]
                                              : ", {"
                                        : "";
                    };

                    static constexpr auto spacer =
                        [](auto r, auto const& iidxs) constexpr noexcept
                    {
                        return (r + 1 == rank) ? (iidxs[rank - 1] + 1 ==
                                                  patch_shape_t::size(rank - 1))
                                                     ? "}\n"
                                                     : "}"
                                               : ", ";
                    };
                    [this, &tree, &h, i, &idxs, &prefix]<std::size_t... I>(
                        std::index_sequence<I...>
                    )
                    {
                        (
                            (m_os
                             << prefix(I, idxs)
                             << tree.template get_patch<std::tuple_element_t<I, map_t>>(i)
                                    .data()[idxs]
                             << spacer(I, idxs)),
                            ...
                        );
                    }(std::make_index_sequence<std::tuple_size_v<map_t>>{});
                }
            );
        }
    }

private:
    static auto print_header(std::ostream& os, auto depth) -> std::ostream&
    {
        for (auto i = decltype(depth){}; i != depth; ++i)
        {
            os << '\t';
        }
        return os;
    }

    std::ostream& m_os;
};

struct vtk_print
{
public:
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
        using tree_t          = std::remove_cvref_t<decltype(tree)>;
        using index_t         = typename tree_t::linear_index_t;
        using patch_index_t   = typename tree_t::patch_index_t;
        using tree_index_t    = typename tree_t::patch_index_t;
        using patch_layout_t  = typename tree_t::patch_layout_t;
        using padded_layout_t = typename patch_layout_t::padded_layout_t;
        using value_type =
            std::tuple_element<0, typename tree_t::deconstructed_raw_map_types_t>;
        using lc_t = patch_layout_t::full_iteration_t;

        constexpr auto dim = std::remove_cvref_t<decltype(tree)>::rank();
        constexpr auto box_points =
            static_cast<index_t>(utility::cx_functions::pow(dim, 2));
        static_assert(dim == 2 || dim == 3);
        const auto tree_size  = tree.size();
        const auto cell_count = tree_size * padded_layout_t::elements();

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

        uint32_t max_coord   = 1u << IndexType::max_depth();
        uint32_t max_coord_y = max_coord * total_size_y;
        auto     patch_id    = tree.get_node_index_at(patch_idx);
        auto     level       = patch_id.level();
        auto     max_depth   = IndexType::max_depth();
        uint32_t patch_size  = 1u << (max_depth - level);

        auto [patch_coords, _] = IndexType::decode(patch_id.id());

        // Base position for this patch (including space for halos)
        uint32_t patch_base_x = total_size_x * patch_coords[0];
        uint32_t patch_base_y = total_size_y * patch_coords[1];
        // TODO: Why do i need to put two ::types here
        file << "POINTS " << cell_count * box_points << ' '
             << type_repr.template operator()<typename value_type::type::type>() << '\n';
        for (std::size_t i = 0; i < tree_size; ++i)
        {
            amr::containers::manipulators::shaped_for<lc_t>(
                [&file](auto const& p, auto const& idxs) { file << p[idxs] << '\n'; }
            );

            uint32_t cell_x = patch_base_x + static_cast<uint32_t>(j) * patch_size;
            uint32_t cell_y = patch_base_y + static_cast<uint32_t>(i) * patch_size;

            // FLIP Y coordinates for top-left origin
            uint32_t flipped_y     = max_coord_y - cell_y - patch_size;
            uint32_t flipped_y_top = max_coord_y - cell_y;

            // Add the 4 corners of this cell (with Y flipped)
            points.push_back({ cell_x, flipped_y_top, 0 });              // top-left
            points.push_back({ cell_x + patch_size, flipped_y_top, 0 }); // top-right
            points.push_back({ cell_x + patch_size, flipped_y, 0 });     // bottom-right
            points.push_back({ cell_x, flipped_y, 0 });                  // bottom-left

            const auto id        = tree.get_node_index_at(i);
            const auto level     = id.level();
            const auto max_depth = tree_index_t::max_depth();
            uint32_t   cell_size = 1u << (max_depth - level);

            auto const& [coords, _] = tree_index_t::decode(id.id());
            auto const x            = coords[0];
            auto const y            = coords[1];

            file << x << ' ' << y << ' ' << 0 << '\n';
            file << x + cell_size << ' ' << y << ' ' << 0 << '\n';
            file << x + cell_size << ' ' << y + cell_size << ' ' << 0 << '\n';
            file << x << ' ' << y + cell_size << ' ' << 0 << '\n';
            if constexpr (dim > 2)
            {
                // TODO: Implement 3D
                auto z = coords[2];
                DEFAULT_SOURCE_LOG_FATAL("3D printer not implemened");
                utility::error_handling::assert_unreachable();
            }
        }

        file << "CELLS " << cell_count << ' ' << cell_count * 5 << '\n';
        for (std::size_t i = 0; i != cell_count * box_points; i += box_points)
        {
            file << box_points;
            for (auto j = index_t{}; j != box_points; ++j)
            {
                file << ' ' << i + j;
            }
            file << '\n';
        }

        file << "CELL_TYPES " << cell_count << '\n';
        for (std::size_t i = 0; i != cell_count; ++i)
        {
            file << "9\n";
        }

        file << "CELL_DATA " << cell_count << '\n';
        file << "SCALARS cell_index int 1\n";
        file << "LOOKUP_TABLE default\n";
        for (std::size_t i = 0; i != cell_count; ++i)
        {
            file << i << '\n';
        }

        file << "CELL_DATA " << cell_count << '\n';
        file << "SCALARS is_halo int 1\n";
        file << "LOOKUP_TABLE default\n";
        for (std::size_t i = 0; i != tree_size; ++i)
        {
            for (auto j = typename padded_layout_t::size_type{};
                 j != padded_layout_t::elements();
                 ++j)
            {
                file << utils::patches::is_halo_cell<patch_layout_t>(j) << '\n';
            }
        }

        [&tree, &file, tree_size]<std::size_t... I>(std::index_sequence<I...>)
        {
            (((void)I,
              [&tree, &file, tree_size]()
              {
                  using element_type = typename std::tuple_element<
                      I,
                      typename tree_t::deconstructed_raw_map_types_t>::type;
                  file << "CELL_DATA " << cell_count << '\n';
                  file << "SCALARS " << element_type::name() << ' '
                       << type_repr.template operator()<typename element_type::type>()
                       << " 1\n";
                  file << "LOOKUP_TABLE default\n";
                  for (size_t i = 0; i != tree_size; ++i)
                  {
                      auto const& patch = tree.template get_patch<element_type>(i);
                      amr::containers::manipulators::for_each<lc_t>(
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

#endif // AMR_INCLUDED_NDT_STRUCTURED_PRINT
