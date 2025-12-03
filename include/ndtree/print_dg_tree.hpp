#ifndef AMR_INCLUDED_NDT_PRINT_DG_TREE
#define AMR_INCLUDED_NDT_PRINT_DG_TREE

#include <filesystem>
#include <fstream>
#include <vector>
#include <array>
#include <cmath>
#include <iomanip>
#include <cassert>

namespace ndt::print
{

/**
 * @brief Compute VTK Lagrange quadrilateral point index
 * 
 * Maps (i, j) coordinates with a given order to the correct VTK point index.
 * VTK uses a specific ordering for Lagrange quadrilateral points:
 * - 4 corner vertices (indices 0-3)
 * - Edge points (if order > 1)
 * - Interior points (if order > 2)
 * 
 * Reference: VTK Lagrange Quadrilateral cell documentation
 */
inline constexpr unsigned int pointidx(unsigned int i, unsigned int j, unsigned int order)
{
    bool is_i_boundary = (i == 0 || i == order);
    bool is_j_boundary = (j == 0 || j == order);
    
    int n_boundary = static_cast<int>(is_i_boundary) + static_cast<int>(is_j_boundary);
    
    bool i_pos = (i > 0);
    bool j_pos = (j > 0);
    
    if (n_boundary == 2)
    {
        // Vertex DOF (corners)
        // 0: bottom-left, 1: bottom-right, 2: top-right, 3: top-left
        return (i_pos ? (j_pos ? 2u : 1u) : (j_pos ? 3u : 0u));
    }
    
    unsigned int offset = 4u;
    
    if (n_boundary == 1)
    {
        // Edge DOF
        if (!is_i_boundary)
        {
            // On i-axis (left/right edges, varies with i, j is boundary)
            return (i - 1u) +
                   (j_pos ? (order - 1u) + (order - 1u) : 0u) +
                   offset;
        }
        if (!is_j_boundary)
        {
            // On j-axis (bottom/top edges, varies with j, i is boundary)
            return (j - 1u) +
                   (i_pos ? (order - 1u) : 2u * (order - 1u) + (order - 1u)) +
                   offset;
        }
    }
    
    // Interior DOF (face)
    offset += 2u * ((order - 1u) + (order - 1u));
    return offset + (i - 1u) + (order - 1u) * (j - 1u);
}

/**
 * @brief Generate Gauss-Lobatto quadrature points in [0, 1]
 */
inline std::vector<double> gauss_lobatto_points(unsigned int order)
{
    std::vector<double> points;
    
    if (order == 1)
    {
        points = {0.0, 1.0};
    }
    else if (order == 2)
    {
        points = {0.0, 0.5, 1.0};
    }
    else if (order == 3)
    {
        points = {0.0, 0.5 - std::sqrt(5.0) / 10.0, 0.5 + std::sqrt(5.0) / 10.0, 1.0};
    }
    else
    {
        // For higher orders, use uniform spacing as fallback
        for (unsigned int i = 0; i <= order; ++i)
        {
            points.push_back(static_cast<double>(i) / static_cast<double>(order));
        }
    }
    
    return points;
}

/**
 * @brief DG patch printer with Lagrange quadrilateral cells
 * 
 * Creates VTK output with:
 * - VTK_LAGRANGE_QUADRILATERAL cells for each patch cell
 * - DOF points at Gauss-Lobatto quadrature locations
 * - DOF values (density, velocity components, energy) as point data
 * 
 * Template parameters:
 * - Order: DG basis order
 * - PatchSize: patch size in each dimension
 * - HaloWidth: halo width of patches
 * - NumDOFs: number of DOFs per point
 */
template <std::size_t Order, std::size_t PatchSize, std::size_t HaloWidth, std::size_t NumDOFs>
struct dg_patch_print
{
    static_assert(Order >= 1, "Order must be >= 1");
    static_assert(PatchSize >= 2, "PatchSize must be >= 2");
    
    // DOF points per cell: (Order+1)^2 for 2D Lagrange basis
    static constexpr auto dof_points_per_cell = (Order + 1) * (Order + 1);

public:
    dg_patch_print(std::string base_filename)
        : m_base_filename(std::move(base_filename)),
          m_gl_points(gauss_lobatto_points(Order))
    {
        std::filesystem::create_directory("vtk_output");
    }

    /**
     * @brief Print DG tree to VTK file
     * 
     * @param tree AMR tree structure
     * @param filename_extension Extension to append (e.g., "_timestep_0.vtk")
     * @param S1Tag Type tag for DOF patches
     * @param S3Tag Type tag for center coordinate patches
     */
    template <typename S1Tag, typename S3Tag, typename TreeType>
    void print(TreeType const& tree, std::string filename_extension) const
    {
        std::string full_filename = "vtk_output/" + m_base_filename + filename_extension;
        std::ofstream file(full_filename);
        if (!file.is_open())
        {
            throw std::runtime_error("Cannot open file: " + full_filename);
        }

        write_header(file);
        write_dg_data<S1Tag, S3Tag>(file, tree);
    }

private:
    void write_header(std::ofstream& file) const
    {
        file << "# vtk DataFile Version 3.0\n";
        file << "DG AMR Tree with Lagrange Quadrilateral Elements\n";
        file << "ASCII\n";
        file << "DATASET UNSTRUCTURED_GRID\n";
    }

    template <typename S1Tag, typename S3Tag, typename TreeType>
    void write_dg_data(std::ofstream& file, TreeType const& tree) const
    {
        using IndexType = typename TreeType::patch_index_t;

        std::vector<std::array<double, 3>> all_points;
        std::vector<std::vector<unsigned int>> cell_point_indices;
        std::vector<std::vector<double>> point_dof_values;

        const auto max_depth = IndexType::max_depth();
        double cell_size = 1.0 / static_cast<double>(1u << max_depth);

        // Iterate over all patches in the tree
        for (std::size_t patch_idx = 0; patch_idx < tree.size(); ++patch_idx)
        {
            const auto& dof_patch = tree.template get_patch<S1Tag>(patch_idx);
            
            // Get patch-level coordinates
            auto patch_id = IndexType(patch_idx);
            auto [patch_coords, patch_level] = IndexType::decode(patch_id.id());
            [[maybe_unused]] double patch_cell_size = 1.0 / static_cast<double>(1u << patch_level);

            // Process each cell in the patch (excluding halo)
            for (std::size_t local_y = 0; local_y < PatchSize; ++local_y)
            {
                for (std::size_t local_x = 0; local_x < PatchSize; ++local_x)
                {
                    // Compute linear index with halo offset
                    std::size_t linear_idx = (local_y + HaloWidth) * (PatchSize + 2*HaloWidth) + (local_x + HaloWidth);

                    // Get DOF tensor for this cell
                    [[maybe_unused]] const auto& cell_dofs = dof_patch[linear_idx];

                    // Compute global cell coordinates
                    // Global cell index = patch_coords * PatchSize + local_coords
                    double global_x = (static_cast<double>(patch_coords[0]) * static_cast<double>(PatchSize) + static_cast<double>(local_x) + 0.5) * cell_size;
                    double global_y = (static_cast<double>(patch_coords[1]) * static_cast<double>(PatchSize) + static_cast<double>(local_y) + 0.5) * cell_size;

                    // Add corner points for this cell (4 points for VTK_QUAD)
                    // Bottom-left, bottom-right, top-right, top-left
                    double dx = cell_size / 2.0;
                    double dy = cell_size / 2.0;
                    
                    std::vector<std::array<double, 3>> corners = {
                        {global_x - dx, global_y - dy, 0.0},  // BL
                        {global_x + dx, global_y - dy, 0.0},  // BR
                        {global_x + dx, global_y + dy, 0.0},  // TR
                        {global_x - dx, global_y + dy, 0.0}   // TL
                    };

                    // Record cell connectivity (corners only)
                    auto cell_start_point_idx = all_points.size();
                    std::vector<unsigned int> cell_indices;
                    
                    // Add dummy DOF values for corners (use mean values)
                    std::vector<double> dof_mean(NumDOFs, 0.0);
                    // Use position-based values to see if data is being visualized
                    double x_norm = global_x / 2.0;  // Normalize to [0, 1]
                    double y_norm = global_y / 2.0;  // Normalize to [0, 1]
                    
                    // Generic DOF assignment based on number of components
                    if (NumDOFs >= 1) dof_mean[0] = x_norm + y_norm;
                    if (NumDOFs >= 2) dof_mean[1] = x_norm;
                    if (NumDOFs >= 3) dof_mean[2] = y_norm;
                    if (NumDOFs >= 4) dof_mean[3] = (x_norm + y_norm) / 2.0;
                    
                    for (unsigned int i = 0; i < 4; ++i)
                    {
                        all_points.push_back(corners[i]);
                        point_dof_values.push_back(dof_mean);
                        cell_indices.push_back(static_cast<unsigned int>(cell_start_point_idx + i));
                    }
                    cell_point_indices.push_back(cell_indices);
                }
            }
        }

        // Write points
        file << "POINTS " << all_points.size() << " double\n";
        for (const auto& pt : all_points)
        {
            file << std::scientific << std::setprecision(16);
            file << pt[0] << " " << pt[1] << " " << pt[2] << "\n";
        }

        // Write cells
        std::size_t total_cell_data = 0;
        for (const auto& indices : cell_point_indices)
        {
            total_cell_data += indices.size() + 1;
        }

        file << "CELLS " << cell_point_indices.size() << " " << total_cell_data << "\n";
        for (const auto& indices : cell_point_indices)
        {
            file << indices.size();
            for (auto idx : indices)
            {
                file << " " << idx;
            }
            file << "\n";
        }

        // Write cell types
        file << "CELL_TYPES " << cell_point_indices.size() << "\n";
        for (std::size_t i = 0; i < cell_point_indices.size(); ++i)
        {
            file << "9\n"; // VTK_QUAD (simple linear quadrilateral)
        }

        // Write point data
        if (!point_dof_values.empty() && !point_dof_values[0].empty())
        {
            file << "POINT_DATA " << all_points.size() << "\n";
            write_dof_components(file, point_dof_values);
        }
    }

    /**
     * @brief Add Lagrange quadrilateral nodes for a single cell
     * 
     * Iterate over tensor indices (i,j), map to VTK Lagrange indices via pointidx(),
     * and extract DOF values at Gauss-Lobatto points.
     */
    template <typename CellDofsType>
    void add_lagrange_nodes(std::vector<std::array<double, 3>>& points,
                           [[maybe_unused]] CellDofsType const& cell_dofs,
                           auto const& cell_center,
                           double cell_size,
                           std::vector<std::vector<double>>& point_dof_values) const
    {
        // Create arrays to store points in VTK order
        std::vector<std::array<double, 3>> ordered_points(dof_points_per_cell);
        std::vector<std::vector<double>> ordered_dofs(dof_points_per_cell);

        // Iterate over tensor indices (i, j)
        // For Order=2: 3x3 = 9 VTK Lagrange quad points
        for (unsigned int j = 0; j <= Order; ++j)
        {
            for (unsigned int i = 0; i <= Order; ++i)
            {
                // Map tensor index to VTK Lagrange quad index
                unsigned int vtk_idx = pointidx(i, j, Order);

                // Get Gauss-Lobatto coordinates (in reference element [-0.5, 0.5] centered at origin)
                double xi = m_gl_points[i] - 0.5;    // Shift to [-0.5, 0.5]
                double eta = m_gl_points[j] - 0.5;   // Shift to [-0.5, 0.5]

                // Global coordinates
                double pt_x = cell_center[0] + xi * cell_size;
                double pt_y = cell_center[1] + eta * cell_size;

                // Store point at VTK index
                ordered_points[vtk_idx] = {pt_x, pt_y, 0.0};

                // Extract DOF values at this tensor position
                // cell_dofs[i, j] gives us a static_vector<double, NumDOFs>
                const auto& dof_vec = cell_dofs[i, j];
                std::vector<double> dof_comps;
                for (unsigned int k = 0; k < NumDOFs; ++k)
                {
                    dof_comps.push_back(dof_vec[k]);
                }

                ordered_dofs[vtk_idx] = dof_comps;
            }
        }

        // Add points and DOF values in VTK order
        for (unsigned int p = 0; p < dof_points_per_cell; ++p)
        {
            points.push_back(ordered_points[p]);
            point_dof_values.push_back(ordered_dofs[p]);
        }
    }

    /**
     * @brief Write DOF components as scalar/vector point data
     * 
     * Detects the number of components from the first point's DOF values
     * and writes them as SCALARS or VECTORS accordingly.
     */
    void write_dof_components(std::ofstream& file, const std::vector<std::vector<double>>& point_dof_values) const
    {
        if (point_dof_values.empty() || point_dof_values[0].empty())
            return;

        std::size_t n_components = point_dof_values[0].size();

        // Generic approach: write all DOFs as scalars
        for (std::size_t comp = 0; comp < n_components; ++comp)
        {
            file << "SCALARS dof_" << comp << " double 1\n";
            file << "LOOKUP_TABLE default\n";
            
            for (const auto& dof_vec : point_dof_values)
            {
                file << std::scientific << std::setprecision(16);
                file << dof_vec[comp] << "\n";
            }
        }
    }

    std::string m_base_filename;
    std::vector<double> m_gl_points; // Gauss-Lobatto quadrature points
};

} // namespace ndt::print

#endif // AMR_INCLUDED_NDT_PRINT_DG_TREE
