#ifndef AMR_INCLUDED_NDT_PRINT_DG_TREE_2D
#define AMR_INCLUDED_NDT_PRINT_DG_TREE_2D

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <string>
#include <vector>

namespace ndt::print
{

// Byte swap for big-endian output (VTK binary format requirement)
template <typename T>
inline T swap_endian(T value)
{
    if constexpr (sizeof(T) == 4)
    {
        uint32_t v;
        std::memcpy(&v, &value, sizeof(T));
        v = ((v & 0xFF000000) >> 24) | ((v & 0x00FF0000) >> 8) | ((v & 0x0000FF00) << 8) |
            ((v & 0x000000FF) << 24);
        T result;
        std::memcpy(&result, &v, sizeof(T));
        return result;
    }
    else if constexpr (sizeof(T) == 8)
    {
        uint64_t v;
        std::memcpy(&v, &value, sizeof(T));
        v = ((v & 0xFF00000000000000ULL) >> 56) | ((v & 0x00FF000000000000ULL) >> 40) |
            ((v & 0x0000FF0000000000ULL) >> 24) | ((v & 0x000000FF00000000ULL) >> 8) |
            ((v & 0x00000000FF000000ULL) << 8) | ((v & 0x0000000000FF0000ULL) << 24) |
            ((v & 0x000000000000FF00ULL) << 40) | ((v & 0x00000000000000FFULL) << 56);
        T result;
        std::memcpy(&result, &v, sizeof(T));
        return result;
    }
    return value;
}

template <typename T>
inline void write_binary(std::ofstream& file, T value)
{
    if constexpr (std::endian::native == std::endian::little)
    {
        value = swap_endian(value);
    }
    file.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

// Helper function to map (i, j) to VTK Lagrange quadrilateral point index
// Following VTK's Lagrange quad ordering: vertices, then edge points, then face points
inline std::size_t
    point_idx_lagrange_quad(std::size_t i, std::size_t j, std::size_t order)
{
    bool is_i_boundary = (i == 0 || i == order);
    bool is_j_boundary = (j == 0 || j == order);
    int  n_boundary = static_cast<int>(is_i_boundary) + static_cast<int>(is_j_boundary);

    bool i_pos = i > 0;
    bool j_pos = j > 0;

    if (n_boundary == 2)
    {
        // Vertex DOF (4 corners)
        return (i_pos ? (j_pos ? 2 : 1) : (j_pos ? 3 : 0));
    }

    std::size_t offset = 4;
    if (n_boundary == 1)
    {
        // Edge DOF
        if (!is_i_boundary)
        {
            // On i-axis (bottom or top edge)
            return (i - 1) + (j_pos ? (order - 1) + (order - 1) : 0) + offset;
        }
        if (!is_j_boundary)
        {
            // On j-axis (left or right edge)
            return (j - 1) + (i_pos ? (order - 1) : 2 * (order - 1) + (order - 1)) +
                   offset;
        }
    }

    // Interior points (face DOF)
    offset += 2 * ((order - 1) + (order - 1));
    return offset + (i - 1) + (order - 1) * (j - 1);
}

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
        std::ofstream file(full_filename, std::ios::binary);
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
        file << "BINARY\n";
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

        for (std::size_t patch_idx = 0; patch_idx < tree.size(); ++patch_idx)
        {
            // Get the actual patch ID from the tree using the proper tree accessor
            auto patch_id = tree.get_node_index_at(patch_idx);

            // Now get the patch using the actual linear index that corresponds to this
            // patch_id The tree stores patches internally, and we need to get the right
            // one
            const auto& dof_patch   = tree.template get_patch<S1Tag>(patch_idx);
            auto        patch_level = patch_id.level();
            auto [patch_coords, _]  = IndexType::decode(patch_id.id());

            // At level L, patch_coords from decode() are already in finest-level grid
            // space They represent the position in a 2^max_depth × 2^max_depth grid Each
            // patch spans 2^(max_depth - L) finest cells
            std::size_t patch_scale = 1u << (max_depth - patch_level);

            // Patch base position is directly from decoded coordinates (already in finest
            // grid space)
            std::size_t max_grid_coord = 1u << max_depth;
            std::size_t patch_base_x_grid =
                std::min(static_cast<std::size_t>(patch_coords[0]), max_grid_coord - 1);
            std::size_t patch_base_y_grid =
                std::min(static_cast<std::size_t>(patch_coords[1]), max_grid_coord - 1);

            // Size of each cell in grid coordinates
            double cell_size_grid =
                static_cast<double>(patch_scale) / static_cast<double>(PatchSize);

            for (std::size_t local_y = 0; local_y < PatchSize; ++local_y)
            {
                for (std::size_t local_x = 0; local_x < PatchSize; ++local_x)
                {
                    std::size_t linear_idx =
                        (local_y + HaloWidth) * padded_patch_size + (local_x + HaloWidth);

                    const auto& cell_dofs = dof_patch[linear_idx];

                    // Cell center in grid coordinates
                    double cell_x_grid =
                        static_cast<double>(patch_base_x_grid) +
                        (static_cast<double>(local_x) + 0.5) * cell_size_grid;
                    double cell_y_grid =
                        static_cast<double>(patch_base_y_grid) +
                        (static_cast<double>(local_y) + 0.5) * cell_size_grid;

                    // Normalize to [0,1] physical domain
                    double normalization_factor = static_cast<double>(1u << max_depth);
                    double cell_x               = cell_x_grid / normalization_factor;
                    double cell_y               = cell_y_grid / normalization_factor;
                    double cell_h               = cell_size_grid / normalization_factor;

                    add_cell_2d(
                        cell_dofs, cell_x, cell_y, cell_h, points, cells, point_dofs
                    );
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
        constexpr std::size_t Np = Order + 1;

        // Create (Order+1) x (Order+1) points in VTK Lagrange quad order
        std::vector<unsigned int> conn;
        conn.reserve(Np * Np);

        // Build points in VTK Lagrange quad ordering
        for (std::size_t jj = 0; jj <= Order; ++jj)
        {
            for (std::size_t ii = 0; ii <= Order; ++ii)
            {
                double xi  = double(ii) / double(Order);
                double eta = double(jj) / double(Order);

                // map to physical coordinates
                double x = cx + (2.0 * xi - 1.0) * (0.5 * h);
                double y = cy + (2.0 * eta - 1.0) * (0.5 * h);

                unsigned int pid = static_cast<unsigned int>(points.size());
                points.push_back({ x, y, 0.0 });

                // Get the correct VTK index for this (i, j) position
                std::size_t vtk_idx = point_idx_lagrange_quad(ii, jj, Order);

                // Ensure conn is large enough
                if (conn.size() <= vtk_idx)
                {
                    conn.resize(vtk_idx + 1);
                }

                conn[vtk_idx] = pid;

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
                    // Binary VTK handles NaN/Inf natively - no conversion needed
                }

                point_dofs.push_back(vals);
            }
        }

        cells.push_back(conn);
    }

    void write_points(
        std::ofstream&                            file,
        const std::vector<std::array<double, 3>>& points
    ) const
    {
        file << "POINTS " << points.size() << " double\n";
        for (const auto& pt : points)
        {
            write_binary(file, pt[0]);
            write_binary(file, pt[1]);
            write_binary(file, pt[2]);
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
            write_binary(file, static_cast<int>(c.size()));
            for (auto pt : c)
                write_binary(file, static_cast<int>(pt));
        }
    }

    void write_cell_types(std::ofstream& file, std::size_t n) const
    {
        file << "CELL_TYPES " << n << "\n";
        for (std::size_t i = 0; i < n; ++i)
            write_binary(file, static_cast<int>(70)); // VTK_LAGRANGE_QUADRILATERAL
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
                double value = (comp < vals.size()) ? vals[comp] : 0.0;
                write_binary(file, value);
            }
        }
    }

    std::string m_base_filename;
};

} // namespace ndt::print

#endif // AMR_INCLUDED_NDT_PRINT_DG_TREE_2D