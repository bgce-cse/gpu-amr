#ifndef AMR_INCLUDED_NDT_PRINT_DG_TREE_2D
#define AMR_INCLUDED_NDT_PRINT_DG_TREE_2D

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <string>
#include <vector>

namespace ndt::print
{

template <typename GlobalConfigType, typename Policy>
struct dg_tree_printer_2d
{
    static constexpr std::size_t Order     = Policy::Order;
    static constexpr std::size_t PatchSize = Policy::PatchSize;
    static constexpr std::size_t HaloWidth = Policy::HaloWidth;
    static constexpr std::size_t NumDOFs   = Policy::DOFs;

    static constexpr auto padded_patch_size = PatchSize + 2 * HaloWidth;

    explicit dg_tree_printer_2d(std::string base_filename)
        : m_base_filename(std::move(base_filename))
    {
        std::filesystem::create_directory("vtk_output");
    }

    template <typename S1Tag, typename TreeType>
    void print(TreeType const& tree, std::string filename_extension) const
    {
        std::string full_filename = "vtk_output/" + m_base_filename + "_Order" +
                                    std::to_string(Order) + filename_extension;
        std::ofstream file(full_filename);
        if (!file.is_open())
            throw std::runtime_error("Cannot open file: " + full_filename);

        write_header(file);
        write_dg_data_2d<S1Tag>(file, tree);
    }

private:
    void write_header(std::ofstream& file) const
    {
        file << "# vtk DataFile Version 3.0\n";
        file << "DG 2D Printer (Order=" << Order << ")\n";
        file << "ASCII\n";
        file << "DATASET UNSTRUCTURED_GRID\n";
    }

    template <typename S1Tag, typename TreeType>
    void write_dg_data_2d(std::ofstream& file, TreeType const& tree) const
    {
        using IndexType = typename TreeType::patch_index_t;

        std::vector<std::array<double, 3>>     points;
        std::vector<std::vector<unsigned int>> cells;
        std::vector<std::vector<double>>       point_dofs;

        const auto max_depth = IndexType::max_depth();
        double     cell_size = 1.0 / static_cast<double>(1u << max_depth);

        for (std::size_t patch_idx = 0; patch_idx < tree.size(); ++patch_idx)
        {
            const auto& dof_patch = tree.template get_patch<S1Tag>(patch_idx);

            auto patch_id                    = IndexType(patch_idx);
            auto [patch_coords, patch_level] = IndexType::decode(patch_id.id());

            for (std::size_t local_y = 0; local_y < PatchSize; ++local_y)
            {
                for (std::size_t local_x = 0; local_x < PatchSize; ++local_x)
                {
                    std::size_t linear_idx =
                        (local_y + HaloWidth) * padded_patch_size + (local_x + HaloWidth);

                    const auto& cell_dofs = dof_patch[linear_idx];

                    double cx = (static_cast<double>(patch_coords[0]) * PatchSize +
                                 static_cast<double>(local_x) + 0.5) *
                                cell_size;
                    double cy = (static_cast<double>(patch_coords[1]) * PatchSize +
                                 static_cast<double>(local_y) + 0.5) *
                                cell_size;

                    add_cell_2d(cell_dofs, cx, cy, cell_size, points, cells, point_dofs);
                }
            }
        }

        write_points(file, points);
        write_cells(file, cells);
        write_cell_types(file, cells.size());
        write_point_data(file, point_dofs);
    }

    template <typename CellDofsType>
    void add_cell_2d(
        CellDofsType const&                     cell_dofs,
        double                                  cx,
        double                                  cy,
        double                                  h,
        std::vector<std::array<double, 3>>&     points,
        std::vector<std::vector<unsigned int>>& cells,
        std::vector<std::vector<double>>&       point_dofs
    ) const
    {
        // constexpr std::size_t Np = Order + 1;

        // Create sub-quads from the high-order element for better visualization
        // This breaks each DG cell into Order x Order linear quads
        for (std::size_t jj = 0; jj < Order; ++jj)
        {
            for (std::size_t ii = 0; ii < Order; ++ii)
            {
                std::vector<unsigned int> conn;
                conn.reserve(4);

                // Four corners of this sub-quad in reference space
                std::array<std::array<std::size_t, 2>, 4> corners = {
                    {
                     { ii, jj },         // bottom-left
                        { ii + 1, jj },     // bottom-right
                        { ii + 1, jj + 1 }, // top-right
                        { ii, jj + 1 }      // top-left
                    }
                };

                for (const auto& [i, j] : corners)
                {
                    double xi  = double(i) / double(Order);
                    double eta = double(j) / double(Order);

                    // map to physical coordinates
                    double x = cx + (2.0 * xi - 1.0) * (0.5 * h);
                    double y = cy + (2.0 * eta - 1.0) * (0.5 * h);

                    unsigned int pid = static_cast<unsigned int>(points.size());
                    points.push_back({ x, y, 0.0 });
                    conn.push_back(pid);

                    typename GlobalConfigType::Basis::vector_t pos;
                    pos[0] = xi;
                    pos[1] = eta;

                    auto value = GlobalConfigType::Basis::evaluate_basis(cell_dofs, pos);

                    std::vector<double> vals(NumDOFs, 0.0);
                    for (std::size_t k = 0;
                         k < std::min<std::size_t>(NumDOFs, value.flat_size());
                         ++k)
                    {
                        vals[k] = value[k];
                    }
                    point_dofs.push_back(vals);
                }

                cells.push_back(conn);
            }
        }
    }

    void write_points(
        std::ofstream&                            file,
        const std::vector<std::array<double, 3>>& points
    ) const
    {
        file << "POINTS " << points.size() << " double\n";
        for (const auto& pt : points)
        {
            file << std::scientific << std::setprecision(16) << pt[0] << " " << pt[1]
                 << " " << pt[2] << "\n";
        }
    }

    void write_cells(
        std::ofstream&                                file,
        const std::vector<std::vector<unsigned int>>& cells
    ) const
    {
        std::size_t total = 0;
        for (const auto& c : cells)
            total += c.size() + 1;

        file << "CELLS " << cells.size() << " " << total << "\n";
        for (const auto& c : cells)
        {
            file << c.size();
            for (auto pt : c)
                file << " " << pt;
            file << "\n";
        }
    }

    void write_cell_types(std::ofstream& file, std::size_t n) const
    {
        file << "CELL_TYPES " << n << "\n";
        for (std::size_t i = 0; i < n; ++i)
            file << 9 << "\n"; // VTK_QUAD (standard linear quad)
    }

    void write_point_data(
        std::ofstream&                          file,
        const std::vector<std::vector<double>>& point_dofs
    ) const
    {
        if (point_dofs.empty()) return;
        file << "POINT_DATA " << point_dofs.size() << "\n";

        for (std::size_t comp = 0; comp < NumDOFs; ++comp)
        {
            file << "SCALARS dof_component_" << comp << " double 1\n";
            file << "LOOKUP_TABLE default\n";
            for (const auto& vals : point_dofs)
            {
                if (comp < vals.size())
                    file << std::scientific << std::setprecision(16) << vals[comp]
                         << "\n";
            }
        }
    }

    std::string m_base_filename;
};

} // namespace ndt::print

#endif // AMR_INCLUDED_NDT_PRINT_DG_TREE_2D