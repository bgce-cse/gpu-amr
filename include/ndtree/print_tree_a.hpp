#ifndef AMR_INCLUDED_NDT_STRUCTURED_PRINT
#define AMR_INCLUDED_NDT_STRUCTURED_PRINT

#include "morton/morton_id.hpp"
#include "ndtree.hpp"
#include <algorithm>
#include <bitset>
#include <filesystem>
#include <fstream>
#include <ostream>
#include <ranges>

namespace ndt::print
{

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
        std::vector<size_t>                  cell_indices;

        // For flat_ndtree: iterate over all valid indices

        for (size_t i = 0; i < tree.size(); ++i)
        {
            auto     id        = tree.get_node_index_at(i); // or use a public accessor
            auto     level     = id.level();
            auto     max_depth = IndexType::max_depth();
            uint32_t cell_size = 1u << (max_depth - level);

            auto [coords, _] = IndexType::decode(id.id());
            uint32_t x       = coords[0];
            uint32_t y       = coords[1];

            cell_indices.push_back(points.size());
            points.push_back({ x, y, 0 });
            points.push_back({ x + cell_size, y, 0 });
            points.push_back({ x + cell_size, y + cell_size, 0 });
            points.push_back({ x, y + cell_size, 0 });
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
            file << i << "\n";  // Just output the cell index
        }
        
        // Write cell data - S1 component (float)
        file << "SCALARS S1_value float 1\n";
        file << "LOOKUP_TABLE default\n";
        for (size_t i = 0; i < cell_indices.size(); ++i)
        {
            auto result = tree.gather_node(i);
            file << std::get<0>(result.data_tuple()).value << "\n";  // First component
        }
        
        // Write S2 component (int)
        file << "SCALARS S2_value int 1\n";
        file << "LOOKUP_TABLE default\n";
        for (size_t i = 0; i < cell_indices.size(); ++i)
        {
            auto result = tree.gather_node(i);
            file << std::get<1>(result.data_tuple()).value << "\n";  // Second component
        }
    }

    std::string m_base_filename;
};

} // namespace ndt::print

#endif // AMR_INCLUDED_NDT_STRUCTURED_PRINT
