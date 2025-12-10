#ifndef AMR_INCLUDED_NDT_STRUCTURED_PRINT
#define AMR_INCLUDED_NDT_STRUCTURED_PRINT

#include "containers/container_manipulations.hpp"
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
        using lc_t          = amr::containers::control::
            loop_control<patch_shape_t, index_t{}, patch_shape_t::sizes(), index_t{ 1 }>;
        static constexpr auto rank = patch_shape_t::rank();

        for (auto i = 0uz; i != tree.size(); ++i)
        {
            using tree_index_t = typename tree_t::patch_index_t;
            using map_t        = typename tree_t::deconstructed_raw_map_types_t;

            auto const h = tree.get_node_index_at(i);

            print_header(m_os, tree_index_t::level(h))
                << "h: " << std::bitset<tree_index_t::bits()>(h.id()).to_string()
                << ", offset: " << decltype(h)::offset_of(h) << '\n';

            containers::manipulators::shaped_for<lc_t>(
                [this, &tree, &h, i](auto const& idxs)
                {
                    static auto prefix = [&h](auto r, auto const& iidxs) constexpr noexcept
                    {
                        // TODO: This was copied from tensor ostream
                        // Write a function that prints a tuple of tensors,
                        // which is what we need here
                        static constexpr auto prefixes = []
                        {
                            static constexpr auto size        = tree_index_t::max_depth();
                            static constexpr decltype(size) k = 2;
                            static constexpr auto           pool = [] constexpr -> auto
                            {
                                std::array<char, size + k> arr{};
                                std::ranges::fill(arr, '\t');
                                arr[size]     = ' ';
                                arr[size + 1] = '{';
                                return arr;
                            }();
                            std::array<std::string_view, size> arr{};
                            for (std::size_t d = 0; d != size; ++d)
                            {
                                arr[d] = std::string_view(pool.data() + size - d, d + k);
                            }
                            return arr;
                        }();
                        return (r == 0) ? (iidxs[rank - 1] == 0)
                                              ? prefixes[tree_index_t::level(h)]
                                              : ", {"
                                        : "";
                    };

                    static constexpr auto spacer =
                        [](auto r, auto const& iidxs) constexpr noexcept
                    {
                        return (r + 1 == rank)
                                   ? (iidxs[rank - 1] + 1 == patch_shape_t::size(rank - 1))
                                         ? "}\n"
                                         : "}"
                                   : ", ";
                    };
                    [this, &tree, &h, i, &idxs]<std::size_t... I>(
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
    vtk_print(std::string base_filename)
        : m_base_filename(std::move(base_filename))
    {
        // Ensure output directory exists
        std::filesystem::create_directory("vtk_output");
    }

    void print(auto const& tree, std::string filename_extension) const
    {
        // Compose full path: ./vtk_output/base_filename + extension
        std::string full_filename = "vtk_output/" + m_base_filename + filename_extension;
        std::ofstream file(full_filename);
        if (!file.is_open())
        {
            throw std::runtime_error("Cannot open file: " + full_filename);
        }
        write_header(file);
        write_points(file, tree);
    }

private:
    void write_header(std::ofstream& file) const
    {
        file << "# vtk DataFile Version 3.0\n";
        file << "AMR Tree Structure\n";
        file << "ASCII\n";
        file << "DATASET UNSTRUCTURED_GRID\n";
    }

    void write_points(std::ofstream& file, auto const& tree) const
    {
        using TreeType  = std::remove_cvref_t<decltype(tree)>;
        using IndexType = typename TreeType::node_index_t;

        std::vector<std::array<uint32_t, 3>> points;
        std::vector<size_t> cell_indices; // store starting index of each cell

        for (auto const& [id, _, ptr] : tree.blocks())
        {
            auto     level           = id.level();
            auto     max_depth       = IndexType::max_depth();
            uint32_t child_cell_size = 1u << (max_depth - level - 1);

            for (typename IndexType::offset_t off = 0; off < 4; off++)
            {
                auto child                 = IndexType::child_of(id, off);
                auto [coords, child_level] = IndexType::decode(child.id());
                uint32_t x                 = coords[0];
                uint32_t y                 = coords[1];

                // Store index of first point for this cell
                cell_indices.push_back(points.size());

                // Add 4 corner points for this child (quad)
                points.push_back({ x, y, 0 });                   // 0: Bottom-left
                points.push_back({ x + child_cell_size, y, 0 }); // 1: Bottom-right
                points.push_back(
                    { x + child_cell_size, y + child_cell_size, 0 }
                );                                               // 2: Top-right
                points.push_back({ x, y + child_cell_size, 0 }); // 3: Top-left
            }
        }

        // Write points
        file << "POINTS " << points.size() << " double\n";
        for (auto const& [x, y, z] : points)
        {
            file << x << " " << y << " " << z << "\n";
        }

        // Write cells (each cell is a quad, 4 points)
        file << "CELLS " << cell_indices.size() << " " << cell_indices.size() * 5 << "\n";
        for (size_t i = 0; i < cell_indices.size(); ++i)
        {
            size_t idx = cell_indices[i];
            file << "4 " << idx << " " << idx + 1 << " " << idx + 2 << " " << idx + 3
                 << "\n";
        }

        // Write cell types (VTK_QUAD = 9)
        file << "CELL_TYPES " << cell_indices.size() << "\n";
        for (size_t i = 0; i < cell_indices.size(); ++i)
        {
            file << "9\n";
        }

        // Write dummy cell data (cell index as scalar)
        file << "CELL_DATA " << cell_indices.size() << "\n";
        file << "SCALARS cell_index int 1\n";
        file << "LOOKUP_TABLE default\n";
        for (size_t i = 0; i < cell_indices.size(); ++i)
        {
            file << i << "\n";
        }
    }

    std::string m_base_filename;
};

} // namespace amr::ndt::print

#endif // AMR_INCLUDED_NDT_STRUCTURED_PRINT
