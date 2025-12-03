#ifndef AMR_INCLUDED_NDT_PRINT_DG_TREE_ADVANCED
#define AMR_INCLUDED_NDT_PRINT_DG_TREE_ADVANCED

#include "containers/static_layout.hpp"
#include "containers/static_shape.hpp"
#include "containers/static_tensor.hpp"
#include "containers/static_vector.hpp"
#include "dg_helpers/basis.hpp"
#include "utility/constexpr_functions.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <numeric>
#include <sstream>
#include <tuple>
#include <vector>

namespace ndt::print
{

/**
 * @brief Helper to compute Gauss-Legendre points directly
 *
 * For a DG basis stored with Order points per dimension (not Order+1),
 * we create GL points for Order quadrature nodes.
 */
template <std::size_t Order>
class GLPointsHelper
{
public:
    explicit GLPointsHelper(double start = 0.0, double end = 1.0)
    {
        // Create GL rule with Order points
        amr::Basis::GaussLegendre<Order> gl(start, end);
        const auto&                      pts = gl.points();

        // Copy points to our vector
        for (unsigned int i = 0; i < Order; ++i)
        {
            m_points[i] = pts[i];
        }
    }

    const amr::containers::static_vector<double, Order>& get_points() const
    {
        return m_points;
    }

private:
    amr::containers::static_vector<double, Order> m_points;
};

/**
 * @brief Advanced DG tree printer with full DOF extraction and point deduplication
 *
 * Creates VTK output with:
 * - Support for arbitrary physical dimensions (2D, 3D)
 * - Support for arbitrary DG orders
 * - Proper extraction of DOF tensor values at Gauss-Legendre quadrature points
 * - Per-component scalar fields for all DOF components
 * - Full cell connectivity with proper VTK cell types
 * - Uses Gauss-Legendre points from the Basis class
 * - Deduplicates points shared between adjacent cells for continuous visualization
 *
 * Template parameters:
 * - Dim: Physical dimension (must match data structure)
 * - Order: DG basis order (must match data structure)
 * - PatchSize: Patch size in each dimension
 * - HaloWidth: Halo width of patches
 * - NumDOFs: Number of DOF components
 */
template <
    std::size_t Dim,
    std::size_t Order,
    std::size_t PatchSize,
    std::size_t HaloWidth,
    std::size_t NumDOFs>
struct dg_tree_printer_advanced;

// ============================================================================
// Specialization for 2D
// ============================================================================

template <
    std::size_t Order,
    std::size_t PatchSize,
    std::size_t HaloWidth,
    std::size_t NumDOFs>
struct dg_tree_printer_advanced<2, Order, PatchSize, HaloWidth, NumDOFs>
{
    static_assert(Order >= 1, "Order must be >= 1");
    static_assert(PatchSize >= 2, "PatchSize must be >= 2");

    static constexpr std::size_t Dim                 = 2;
    static constexpr auto        dof_points_per_cell = Order * Order;
    static constexpr auto        padded_patch_size   = PatchSize + 2 * HaloWidth;

public:
    explicit dg_tree_printer_advanced(std::string base_filename)
        : m_base_filename(std::move(base_filename))
        , m_gl_helper(0.0, 1.0)
    {
        std::filesystem::create_directory("vtk_output");
    }

    template <typename S1Tag, typename TreeType>
    void print(TreeType const& tree, std::string filename_extension) const
    {
        // Include Order in the filename for clarity
        std::string full_filename = "vtk_output/" + m_base_filename + "_Order" +
                                    std::to_string(Order) + filename_extension;
        std::ofstream file(full_filename);
        if (!file.is_open())
        {
            throw std::runtime_error("Cannot open file: " + full_filename);
        }

        write_header(file);
        write_dg_data_2d<S1Tag>(file, tree);
    }

    /**
     * @brief Print debug information for all cells in the tree
     *
     * Extracts and displays the first DOF value for each cell.
     */
    template <typename S1Tag, typename TreeType>
    void print_debug_info(TreeType const& tree) const
    {
        for (std::size_t patch_idx = 0; patch_idx < tree.size(); ++patch_idx)
        {
            const auto& dof_patch = tree.template get_patch<S1Tag>(patch_idx);
            std::cout << "Patch " << patch_idx << ":\n";

            for (std::size_t local_y = 0; local_y < PatchSize; ++local_y)
            {
                for (std::size_t local_x = 0; local_x < PatchSize; ++local_x)
                {
                    std::size_t linear_idx =
                        (local_y + HaloWidth) * (PatchSize + 2 * HaloWidth) +
                        (local_x + HaloWidth);

                    // Get the DOF tensor for this cell
                    const auto& cell_dofs = dof_patch[linear_idx];

                    // Extract DOF value from first tensor node
                    const auto& dof_vec = cell_dofs[0, 0];

                    std::cout << "  Cell [" << local_x << "," << local_y << "]: ";
                    for (unsigned int d = 0; d < NumDOFs; ++d)
                    {
                        if (d < dof_vec.flat_size())
                        {
                            std::cout << std::scientific << std::setprecision(4)
                                      << dof_vec[d] << " ";
                        }
                    }
                    std::cout << "\n";
                }
            }
            std::cout << "\n";
        }
    }

private:
    void write_header(std::ofstream& file) const
    {
        file << "# vtk DataFile Version 3.0\n";
        file << "DG AMR Tree - Advanced Printer 2D (Order=" << Order
             << ", DOFs=" << NumDOFs << ")\n";
        file << "ASCII\n";
        file << "DATASET UNSTRUCTURED_GRID\n";
    }

    template <typename S1Tag, typename TreeType>
    void write_dg_data_2d(std::ofstream& file, TreeType const& tree) const
    {
        using IndexType = typename TreeType::patch_index_t;

        std::vector<std::array<double, 3>>                            all_points;
        std::map<std::pair<unsigned int, unsigned int>, unsigned int> point_map;
        std::vector<std::vector<unsigned int>>                        cell_connectivity;
        std::vector<std::vector<double>>                              all_point_dofs;

        const auto max_depth = IndexType::max_depth();
        double     cell_size = 1.0 / static_cast<double>(1u << max_depth);

        const auto& gl_points = m_gl_helper.get_points();

        // Iterate over all patches
        for (std::size_t patch_idx = 0; patch_idx < tree.size(); ++patch_idx)
        {
            const auto& dof_patch = tree.template get_patch<S1Tag>(patch_idx);

            auto patch_id                    = IndexType(patch_idx);
            auto [patch_coords, patch_level] = IndexType::decode(patch_id.id());

            // Process each cell in the patch
            for (std::size_t local_y = 0; local_y < PatchSize; ++local_y)
            {
                for (std::size_t local_x = 0; local_x < PatchSize; ++local_x)
                {
                    std::size_t linear_idx =
                        (local_y + HaloWidth) * padded_patch_size + (local_x + HaloWidth);
                    const auto& cell_dofs = dof_patch[linear_idx];

                    double global_x = (static_cast<double>(patch_coords[0]) *
                                           static_cast<double>(PatchSize) +
                                       static_cast<double>(local_x) + 0.5) *
                                      cell_size;
                    double global_y = (static_cast<double>(patch_coords[1]) *
                                           static_cast<double>(PatchSize) +
                                       static_cast<double>(local_y) + 0.5) *
                                      cell_size;

                    std::size_t cell_x = patch_coords[0] * PatchSize + local_x;
                    std::size_t cell_y = patch_coords[1] * PatchSize + local_y;

                    add_cell_2d(
                        cell_dofs,
                        global_x,
                        global_y,
                        cell_size,
                        gl_points,
                        cell_x,
                        cell_y,
                        all_points,
                        point_map,
                        cell_connectivity,
                        all_point_dofs
                    );
                }
            }
        }

        write_points(file, all_points);
        write_cells(file, cell_connectivity);
        write_cell_types(file, cell_connectivity.size());
        write_point_data(file, all_point_dofs);
    }

    /**
     * @brief Add DOF points for a 2D cell with point deduplication using grid indices
     */
    template <typename CellDofsType, typename GLPoints>
    void add_cell_2d(
        CellDofsType const&                                            cell_dofs,
        double                                                         cell_center_x,
        double                                                         cell_center_y,
        double                                                         cell_size,
        const GLPoints&                                                gl_points,
        std::size_t                                                    cell_x,
        std::size_t                                                    cell_y,
        std::vector<std::array<double, 3>>&                            all_points,
        std::map<std::pair<unsigned int, unsigned int>, unsigned int>& point_map,
        std::vector<std::vector<unsigned int>>&                        cell_connectivity,
        std::vector<std::vector<double>>&                              all_point_dofs
    ) const
    {
        // For Order=1, we have only 1 point per cell (the center)
        // For proper visualization, we need to create corner points
        if constexpr (Order == 1)
        {
            // Compute cell boundaries
            double half_cell = cell_size * 0.5;
            double x_min     = cell_center_x - half_cell;
            double x_max     = cell_center_x + half_cell;
            double y_min     = cell_center_y - half_cell;
            double y_max     = cell_center_y + half_cell;

            // Corner coordinates: (x_min,y_min), (x_max,y_min), (x_max,y_max),
            // (x_min,y_max)
            std::array<std::pair<double, double>, 4> corners = {
                {
                 { x_min, y_min }, // (0,0)
                    { x_max, y_min }, // (1,0)
                    { x_max, y_max }, // (1,1)
                    { x_min, y_max }  // (0,1)
                }
            };

            std::vector<unsigned int> connectivity;

            // Corner indices based on cell position
            for (std::size_t corner_idx = 0; corner_idx < 4; ++corner_idx)
            {
                // Create grid key based on cell position and corner
                // For Order=1, each cell corner is unique, indexed by (2*cell_x + ix,
                // 2*cell_y + iy)
                std::size_t ix       = corner_idx % 2; // 0 or 1
                std::size_t iy       = corner_idx / 2; // 0 or 1
                auto        grid_key = std::make_pair(
                    static_cast<unsigned int>(cell_x * 2 + ix),
                    static_cast<unsigned int>(cell_y * 2 + iy)
                );

                unsigned int pt_idx;
                if (point_map.find(grid_key) != point_map.end())
                {
                    pt_idx = point_map[grid_key];
                }
                else
                {
                    pt_idx              = static_cast<unsigned int>(all_points.size());
                    point_map[grid_key] = pt_idx;
                    all_points.push_back(
                        { corners[corner_idx].first, corners[corner_idx].second, 0.0 }
                    );

                    // Use cell center DOF for all corner points
                    const auto&         dof_vec = cell_dofs[0, 0];
                    std::vector<double> dof_comps(NumDOFs, 0.0);
                    unsigned int        max_k = std::min(
                        static_cast<unsigned int>(NumDOFs),
                        static_cast<unsigned int>(dof_vec.flat_size())
                    );
                    for (unsigned int k = 0; k < max_k; ++k)
                    {
                        dof_comps[k] = dof_vec[k];
                    }
                    all_point_dofs.push_back(dof_comps);
                }

                connectivity.push_back(pt_idx);
            }

            cell_connectivity.push_back(connectivity);
        }
        else
        {
            // For Order >= 2: use GL points with proper corner extraction
            // Collect all point indices
            std::vector<std::vector<unsigned int>> point_grid(
                Order, std::vector<unsigned int>(Order)
            );

            for (std::size_t j = 0; j < Order; ++j)
            {
                for (std::size_t i = 0; i < Order; ++i)
                {
                    // Compute global node indices for visualization mesh
                    std::size_t global_node_x = cell_x * (Order - 1) + i;
                    std::size_t global_node_y = cell_y * (Order - 1) + j;

                    auto grid_key = std::make_pair(
                        static_cast<unsigned int>(global_node_x),
                        static_cast<unsigned int>(global_node_y)
                    );

                    unsigned int pt_idx;
                    if (point_map.find(grid_key) != point_map.end())
                    {
                        pt_idx = point_map[grid_key];
                    }
                    else
                    {
                        pt_idx = static_cast<unsigned int>(all_points.size());
                        point_map[grid_key] = pt_idx;

                        double xi =
                            (gl_points[static_cast<unsigned int>(i)] - 0.5) * cell_size;
                        double eta =
                            (gl_points[static_cast<unsigned int>(j)] - 0.5) * cell_size;
                        double pt_x = cell_center_x + xi;
                        double pt_y = cell_center_y + eta;

                        all_points.push_back({ pt_x, pt_y, 0.0 });

                        const auto&         dof_vec = cell_dofs[i, j];
                        std::vector<double> dof_comps(NumDOFs, 0.0);
                        unsigned int        max_k = std::min(
                            static_cast<unsigned int>(NumDOFs),
                            static_cast<unsigned int>(dof_vec.flat_size())
                        );
                        for (unsigned int k = 0; k < max_k; ++k)
                        {
                            dof_comps[k] = dof_vec[k];
                        }
                        all_point_dofs.push_back(dof_comps);
                    }

                    point_grid[j][i] = pt_idx;
                }
            }

            // Build connectivity: subdivide higher-order cell into (Order-1)^2 bilinear
            // quads This provides smooth visualization using only bilinear rendering
            // Point grid is stored as [j][i], so we connect (j,i), (j,i+1), (j+1,i+1),
            // (j+1,i)

            for (std::size_t j = 0; j < Order - 1; ++j)
            {
                for (std::size_t i = 0; i < Order - 1; ++i)
                {
                    std::vector<unsigned int> quad_connectivity;
                    quad_connectivity.push_back(point_grid[j][i]);         // (i,j)
                    quad_connectivity.push_back(point_grid[j][i + 1]);     // (i+1,j)
                    quad_connectivity.push_back(point_grid[j + 1][i + 1]); // (i+1,j+1)
                    quad_connectivity.push_back(point_grid[j + 1][i]);     // (i,j+1)
                    cell_connectivity.push_back(quad_connectivity);
                }
            }
        }
    }

    void write_points(
        std::ofstream&                            file,
        const std::vector<std::array<double, 3>>& all_points
    ) const
    {
        file << "POINTS " << all_points.size() << " double\n";
        for (const auto& pt : all_points)
        {
            file << std::scientific << std::setprecision(16);
            file << pt[0] << " " << pt[1] << " " << pt[2] << "\n";
        }
    }

    void write_cells(
        std::ofstream&                                file,
        const std::vector<std::vector<unsigned int>>& cell_connectivity
    ) const
    {
        std::size_t total_connectivity = 0;
        for (const auto& cell : cell_connectivity)
        {
            total_connectivity += cell.size() + 1;
        }

        file << "CELLS " << cell_connectivity.size() << " " << total_connectivity << "\n";
        for (const auto& cell : cell_connectivity)
        {
            file << cell.size();
            for (auto pt_idx : cell)
            {
                file << " " << pt_idx;
            }
            file << "\n";
        }
    }

    void write_cell_types(std::ofstream& file, std::size_t num_cells) const
    {
        file << "CELL_TYPES " << num_cells << "\n";
        for (std::size_t i = 0; i < num_cells; ++i)
        {
            file << "9\n"; // VTK_QUAD (bilinear quadrilateral) for each sub-cell
        }
    }

    void write_point_data(
        std::ofstream&                          file,
        const std::vector<std::vector<double>>& all_point_dofs
    ) const
    {
        std::size_t total_points = all_point_dofs.size();

        if (total_points == 0) return;

        file << "POINT_DATA " << total_points << "\n";

        for (std::size_t comp = 0; comp < NumDOFs; ++comp)
        {
            file << "SCALARS dof_component_" << comp << " double 1\n";
            file << "LOOKUP_TABLE default\n";

            for (const auto& point_dofs : all_point_dofs)
            {
                if (comp < point_dofs.size())
                {
                    file << std::scientific << std::setprecision(16);
                    file << point_dofs[comp] << "\n";
                }
            }
        }
    }

    std::string                   m_base_filename;
    mutable GLPointsHelper<Order> m_gl_helper;
};

// ============================================================================
// Specialization for 3D
// ============================================================================

template <
    std::size_t Order,
    std::size_t PatchSize,
    std::size_t HaloWidth,
    std::size_t NumDOFs>
struct dg_tree_printer_advanced<3, Order, PatchSize, HaloWidth, NumDOFs>
{
    static_assert(Order >= 1, "Order must be >= 1");
    static_assert(PatchSize >= 2, "PatchSize must be >= 2");

    static constexpr std::size_t Dim                 = 3;
    static constexpr auto        dof_points_per_cell = Order * Order * Order;
    static constexpr auto        padded_patch_size   = PatchSize + 2 * HaloWidth;

public:
    explicit dg_tree_printer_advanced(std::string base_filename)
        : m_base_filename(std::move(base_filename))
        , m_gl_helper(0.0, 1.0)
    {
        std::filesystem::create_directory("vtk_output");
    }

    template <typename S1Tag, typename TreeType>
    void print(TreeType const& tree, std::string filename_extension) const
    {
        // Include Order in the filename for clarity
        std::string full_filename = "vtk_output/" + m_base_filename + "_Order" +
                                    std::to_string(Order) + filename_extension;
        std::ofstream file(full_filename);
        if (!file.is_open())
        {
            throw std::runtime_error("Cannot open file: " + full_filename);
        }

        write_header(file);
        write_dg_data_3d<S1Tag>(file, tree);
    }

    /**
     * @brief Print debug information for all cells in the tree
     *
     * Extracts and displays the first DOF value for each cell.
     */
    template <typename S1Tag, typename TreeType>
    void print_debug_info(TreeType const& tree) const
    {
        for (std::size_t patch_idx = 0; patch_idx < tree.size(); ++patch_idx)
        {
            const auto& dof_patch = tree.template get_patch<S1Tag>(patch_idx);
            std::cout << "Patch " << patch_idx << ":\n";

            for (std::size_t local_z = 0; local_z < PatchSize; ++local_z)
            {
                for (std::size_t local_y = 0; local_y < PatchSize; ++local_y)
                {
                    for (std::size_t local_x = 0; local_x < PatchSize; ++local_x)
                    {
                        std::size_t stride     = PatchSize + 2 * HaloWidth;
                        std::size_t linear_idx = (local_z + HaloWidth) * stride * stride +
                                                 (local_y + HaloWidth) * stride +
                                                 (local_x + HaloWidth);

                        // Get the DOF tensor for this cell
                        const auto& cell_dofs = dof_patch[linear_idx];

                        // Extract DOF value from first tensor node
                        const auto& dof_vec = cell_dofs[0, 0, 0];

                        std::cout << "  Cell [" << local_x << "," << local_y << ","
                                  << local_z << "]: ";
                        for (unsigned int d = 0; d < NumDOFs; ++d)
                        {
                            if (d < dof_vec.flat_size())
                            {
                                std::cout << std::scientific << std::setprecision(4)
                                          << dof_vec[d] << " ";
                            }
                        }
                        std::cout << "\n";
                    }
                }
            }
            std::cout << "\n";
        }
    }

private:
    void write_header(std::ofstream& file) const
    {
        file << "# vtk DataFile Version 3.0\n";
        file << "DG AMR Tree - Advanced Printer 3D (Order=" << Order
             << ", DOFs=" << NumDOFs << ")\n";
        file << "ASCII\n";
        file << "DATASET UNSTRUCTURED_GRID\n";
    }

    template <typename S1Tag, typename TreeType>
    void write_dg_data_3d(std::ofstream& file, TreeType const& tree) const
    {
        using IndexType = typename TreeType::patch_index_t;

        std::vector<std::array<double, 3>> all_points;
        std::map<std::tuple<unsigned int, unsigned int, unsigned int>, unsigned int>
                                               point_map;
        std::vector<std::vector<unsigned int>> cell_connectivity;
        std::vector<std::vector<double>>       all_point_dofs;

        const auto max_depth = IndexType::max_depth();
        double     cell_size = 1.0 / static_cast<double>(1u << max_depth);

        const auto& gl_points = m_gl_helper.get_points();

        for (std::size_t patch_idx = 0; patch_idx < tree.size(); ++patch_idx)
        {
            const auto& dof_patch = tree.template get_patch<S1Tag>(patch_idx);

            auto patch_id                    = IndexType(patch_idx);
            auto [patch_coords, patch_level] = IndexType::decode(patch_id.id());

            for (std::size_t local_z = 0; local_z < PatchSize; ++local_z)
            {
                for (std::size_t local_y = 0; local_y < PatchSize; ++local_y)
                {
                    for (std::size_t local_x = 0; local_x < PatchSize; ++local_x)
                    {
                        std::size_t linear_idx =
                            (local_z + HaloWidth) * padded_patch_size *
                                padded_patch_size +
                            (local_y + HaloWidth) * padded_patch_size +
                            (local_x + HaloWidth);

                        const auto& cell_dofs = dof_patch[linear_idx];

                        double global_x = (static_cast<double>(patch_coords[0]) *
                                               static_cast<double>(PatchSize) +
                                           static_cast<double>(local_x) + 0.5) *
                                          cell_size;
                        double global_y = (static_cast<double>(patch_coords[1]) *
                                               static_cast<double>(PatchSize) +
                                           static_cast<double>(local_y) + 0.5) *
                                          cell_size;
                        double global_z = (static_cast<double>(patch_coords[2]) *
                                               static_cast<double>(PatchSize) +
                                           static_cast<double>(local_z) + 0.5) *
                                          cell_size;

                        std::size_t cell_x = patch_coords[0] * PatchSize + local_x;
                        std::size_t cell_y = patch_coords[1] * PatchSize + local_y;
                        std::size_t cell_z = patch_coords[2] * PatchSize + local_z;

                        add_cell_3d(
                            cell_dofs,
                            global_x,
                            global_y,
                            global_z,
                            cell_size,
                            gl_points,
                            cell_x,
                            cell_y,
                            cell_z,
                            all_points,
                            point_map,
                            cell_connectivity,
                            all_point_dofs
                        );
                    }
                }
            }
        }

        write_points(file, all_points);
        write_cells(file, cell_connectivity);
        write_cell_types(file, cell_connectivity.size());
        write_point_data(file, all_point_dofs);
    }

    template <typename CellDofsType, typename GLPoints>
    void add_cell_3d(
        CellDofsType const&                 cell_dofs,
        double                              cell_center_x,
        double                              cell_center_y,
        double                              cell_center_z,
        double                              cell_size,
        const GLPoints&                     gl_points,
        std::size_t                         cell_x,
        std::size_t                         cell_y,
        std::size_t                         cell_z,
        std::vector<std::array<double, 3>>& all_points,
        std::map<std::tuple<unsigned int, unsigned int, unsigned int>, unsigned int>&
                                                point_map,
        std::vector<std::vector<unsigned int>>& cell_connectivity,
        std::vector<std::vector<double>>&       all_point_dofs
    ) const
    {
        // For Order=1, we have only 1 point per cell (the center)
        // For proper visualization, we need to create 8 corner points
        if constexpr (Order == 1)
        {
            // Compute cell boundaries
            double half_cell = cell_size * 0.5;
            double x_min     = cell_center_x - half_cell;
            double x_max     = cell_center_x + half_cell;
            double y_min     = cell_center_y - half_cell;
            double y_max     = cell_center_y + half_cell;
            double z_min     = cell_center_z - half_cell;
            double z_max     = cell_center_z + half_cell;

            // 8 corner coordinates: (x_min/max, y_min/max, z_min/max)
            std::array<std::tuple<double, double, double>, 8> corners = {
                {
                 { x_min, y_min, z_min }, // (0,0,0)
                    { x_max, y_min, z_min }, // (1,0,0)
                    { x_max, y_max, z_min }, // (1,1,0)
                    { x_min, y_max, z_min }, // (0,1,0)
                    { x_min, y_min, z_max }, // (0,0,1)
                    { x_max, y_min, z_max }, // (1,0,1)
                    { x_max, y_max, z_max }, // (1,1,1)
                    { x_min, y_max, z_max }  // (0,1,1)
                }
            };

            std::vector<unsigned int> connectivity;

            // Corner indices based on cell position
            for (std::size_t corner_idx = 0; corner_idx < 8; ++corner_idx)
            {
                // Create grid key based on cell position and corner
                // For Order=1, each cell corner is unique, indexed by (2*cell_x + ix,
                // 2*cell_y + iy, 2*cell_z + iz)
                std::size_t ix       = corner_idx % 2;       // 0 or 1
                std::size_t iy       = (corner_idx / 2) % 2; // 0 or 1
                std::size_t iz       = corner_idx / 4;       // 0 or 1
                auto        grid_key = std::make_tuple(
                    static_cast<unsigned int>(cell_x * 2 + ix),
                    static_cast<unsigned int>(cell_y * 2 + iy),
                    static_cast<unsigned int>(cell_z * 2 + iz)
                );

                unsigned int pt_idx;
                if (point_map.find(grid_key) != point_map.end())
                {
                    pt_idx = point_map[grid_key];
                }
                else
                {
                    pt_idx              = static_cast<unsigned int>(all_points.size());
                    point_map[grid_key] = pt_idx;
                    auto [x, y, z]      = corners[corner_idx];
                    all_points.push_back({ x, y, z });

                    // Use cell center DOF for all corner points
                    const auto&         dof_vec = cell_dofs[0, 0, 0];
                    std::vector<double> dof_comps(NumDOFs, 0.0);
                    unsigned int        max_d = std::min(
                        static_cast<unsigned int>(NumDOFs),
                        static_cast<unsigned int>(dof_vec.flat_size())
                    );
                    for (unsigned int d = 0; d < max_d; ++d)
                    {
                        dof_comps[d] = dof_vec[d];
                    }
                    all_point_dofs.push_back(dof_comps);
                }

                connectivity.push_back(pt_idx);
            }

            cell_connectivity.push_back(connectivity);
        }
        else
        {
            // For Order >= 2: use GL points with proper corner extraction
            // Collect all point indices in 3D grid
            std::vector<std::vector<std::vector<unsigned int>>> point_grid(
                Order,
                std::vector<std::vector<unsigned int>>(
                    Order, std::vector<unsigned int>(Order)
                )
            );

            for (std::size_t k_idx = 0; k_idx < Order; ++k_idx)
            {
                for (std::size_t j_idx = 0; j_idx < Order; ++j_idx)
                {
                    for (std::size_t i_idx = 0; i_idx < Order; ++i_idx)
                    {
                        // Compute global node indices for visualization mesh
                        std::size_t global_node_x = cell_x * (Order - 1) + i_idx;
                        std::size_t global_node_y = cell_y * (Order - 1) + j_idx;
                        std::size_t global_node_z = cell_z * (Order - 1) + k_idx;

                        auto grid_key = std::make_tuple(
                            static_cast<unsigned int>(global_node_x),
                            static_cast<unsigned int>(global_node_y),
                            static_cast<unsigned int>(global_node_z)
                        );

                        unsigned int pt_idx;
                        if (point_map.find(grid_key) != point_map.end())
                        {
                            pt_idx = point_map[grid_key];
                        }
                        else
                        {
                            pt_idx = static_cast<unsigned int>(all_points.size());
                            point_map[grid_key] = pt_idx;

                            double xi =
                                (gl_points[static_cast<unsigned int>(i_idx)] - 0.5) *
                                cell_size;
                            double eta =
                                (gl_points[static_cast<unsigned int>(j_idx)] - 0.5) *
                                cell_size;
                            double zeta =
                                (gl_points[static_cast<unsigned int>(k_idx)] - 0.5) *
                                cell_size;

                            double pt_x = cell_center_x + xi;
                            double pt_y = cell_center_y + eta;
                            double pt_z = cell_center_z + zeta;

                            all_points.push_back({ pt_x, pt_y, pt_z });

                            const auto&         dof_vec = cell_dofs[i_idx, j_idx, k_idx];
                            std::vector<double> dof_comps(NumDOFs, 0.0);
                            unsigned int        max_d = std::min(
                                static_cast<unsigned int>(NumDOFs),
                                static_cast<unsigned int>(dof_vec.flat_size())
                            );
                            for (unsigned int d = 0; d < max_d; ++d)
                            {
                                dof_comps[d] = dof_vec[d];
                            }
                            all_point_dofs.push_back(dof_comps);
                        }

                        point_grid[k_idx][j_idx][i_idx] = pt_idx;
                    }
                }
            }

            // Build connectivity: output all GL points for Lagrange visualization
            // Note: point_grid is stored as [k][j][i] (k=z, j=y, i=x)
            // For VTK Lagrange hexahedron, output in row-major order consistent with
            // storage
            std::vector<unsigned int> connectivity;

            // Output all points in row-major order (i varies fastest, then j, then k)
            for (std::size_t k = 0; k < Order; ++k)
            {
                for (std::size_t j = 0; j < Order; ++j)
                {
                    for (std::size_t i = 0; i < Order; ++i)
                    {
                        connectivity.push_back(point_grid[k][j][i]);
                    }
                }
            }

            cell_connectivity.push_back(connectivity);
        }
    }

    void write_points(
        std::ofstream&                            file,
        const std::vector<std::array<double, 3>>& all_points
    ) const
    {
        file << "POINTS " << all_points.size() << " double\n";
        for (const auto& pt : all_points)
        {
            file << std::scientific << std::setprecision(16);
            file << pt[0] << " " << pt[1] << " " << pt[2] << "\n";
        }
    }

    void write_cells(
        std::ofstream&                                file,
        const std::vector<std::vector<unsigned int>>& cell_connectivity
    ) const
    {
        std::size_t total_connectivity = 0;
        for (const auto& cell : cell_connectivity)
        {
            total_connectivity += cell.size() + 1;
        }

        file << "CELLS " << cell_connectivity.size() << " " << total_connectivity << "\n";
        for (const auto& cell : cell_connectivity)
        {
            file << cell.size();
            for (auto pt_idx : cell)
            {
                file << " " << pt_idx;
            }
            file << "\n";
        }
    }

    void write_cell_types(std::ofstream& file, std::size_t num_cells) const
    {
        file << "CELL_TYPES " << num_cells << "\n";
        for (std::size_t i = 0; i < num_cells; ++i)
        {
            if constexpr (Order == 1)
            {
                file << "12\n"; // VTK_HEXAHEDRON (linear hexahedron) for Order=1
            }
            else
            {
                file << "72\n"; // VTK_LAGRANGE_HEXAHEDRON for Order >= 2
            }
        }
    }

    void write_point_data(
        std::ofstream&                          file,
        const std::vector<std::vector<double>>& all_point_dofs
    ) const
    {
        std::size_t total_points = all_point_dofs.size();

        if (total_points == 0) return;

        file << "POINT_DATA " << total_points << "\n";

        for (std::size_t comp = 0; comp < NumDOFs; ++comp)
        {
            file << "SCALARS dof_component_" << comp << " double 1\n";
            file << "LOOKUP_TABLE default\n";

            for (const auto& point_dofs : all_point_dofs)
            {
                if (comp < point_dofs.size())
                {
                    file << std::scientific << std::setprecision(16);
                    file << point_dofs[comp] << "\n";
                }
            }
        }
    }

    std::string                   m_base_filename;
    mutable GLPointsHelper<Order> m_gl_helper;
};

} // namespace ndt::print

#endif // AMR_INCLUDED_NDT_PRINT_DG_TREE_ADVANCED
