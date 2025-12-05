#ifndef AMR_INCLUDED_NDT_PRINT_DG_TREE_FIXED
#define AMR_INCLUDED_NDT_PRINT_DG_TREE_FIXED

#include "dg_helpers/basis.hpp"
#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

struct S1;

namespace ndt::print
{

// Simple 2D DG tree printer following print_tree_a.hpp approach
template <
    std::size_t Order,
    std::size_t PatchSize,
    std::size_t HaloWidth,
    std::size_t NumDOFs,
    typename PatchLayout>
struct dg_tree_printer_simple
{
    static constexpr std::size_t padded_size = PatchSize + 2 * HaloWidth;

    amr::Basis::Basis<Order, 2> basis{ 0.0, 1.0 };
    mutable int                 file_counter = 0;

    dg_tree_printer_simple(std::string base_filename = "dg_tree")
        : m_base_filename(std::move(base_filename))
    {
    }

    // Print function: template over S1Tag (DOF data) and TreeT
    template <typename S1Tag, typename TreeT>
    void print(const TreeT& tree, double /* time */ = 0.0) const
    {
        std::ostringstream fn;
        fn << "vtk_output/" << m_base_filename << "_Order" << Order << "_" << std::setw(6)
           << std::setfill('0') << file_counter << ".vtu";
        std::string filename = fn.str();

        std::cout << "[DG_PRINTER_SIMPLE] Writing " << filename << "\n";

        std::vector<std::array<double, 3>> points;
        std::vector<int>                   connectivity;
        std::vector<int>                   offsets;
        std::vector<int>                   types;
        std::vector<std::vector<double>>   point_dofs; // DOF values per point

        double cell_size = 0.1; // Grid spacing

        // Iterate over patches
        for (std::size_t patch_idx = 0; patch_idx < tree.size(); ++patch_idx)
        {
            const auto& dof_patch = tree.template get_patch<S1Tag>(patch_idx);
            std::size_t flat_size = padded_size * padded_size;

            // Iterate over cells in patch
            for (std::size_t li = 0; li < flat_size; ++li)
            {
                std::size_t local_y = li / padded_size;
                std::size_t local_x = li % padded_size;

                // Skip halo cells
                if (local_x < HaloWidth || local_x >= PatchSize + HaloWidth ||
                    local_y < HaloWidth || local_y >= PatchSize + HaloWidth)
                    continue;

                const auto& cell_dof_tensor = dof_patch[li];

                // Compute grid position
                std::size_t grid_x = patch_idx * PatchSize + (local_x - HaloWidth);
                std::size_t grid_y = local_y - HaloWidth;

                // Cell corners in physical space
                double x_min = static_cast<double>(grid_x) * cell_size;
                double x_max = x_min + cell_size;
                double y_min = static_cast<double>(grid_y) * cell_size;
                double y_max = y_min + cell_size;

                // Corners in reference space: (-1,-1), (1,-1), (1,1), (-1,1)
                const double corner_refs[4][2] = {
                    { -1.0, -1.0 },
                    { 1.0,  -1.0 },
                    { 1.0,  1.0  },
                    { -1.0, 1.0  }
                };
                const std::array<std::array<double, 3>, 4> corner_coords = {
                    { { x_min, y_min, 0.0 },
                     { x_max, y_min, 0.0 },
                     { x_max, y_max, 0.0 },
                     { x_min, y_max, 0.0 } }
                };

                const auto& gl_pts = basis.quadpoints();

                // Add 4 corner points and interpolate DOF at each
                int pt_base = static_cast<int>(points.size());

                for (int corner = 0; corner < 4; ++corner)
                {
                    points.push_back(corner_coords[corner]);

                    // Interpolate DOF at this corner
                    std::vector<double> corner_dofs(NumDOFs, 0.0);
                    double              xi  = corner_refs[corner][0];
                    double              eta = corner_refs[corner][1];

                    // Evaluate basis functions at corner and sum DOF contributions
                    for (std::size_t qj = 0; qj < Order; ++qj)
                    {
                        for (std::size_t qi = 0; qi < Order; ++qi)
                        {
                            // Evaluate Lagrange basis L_i(xi) * L_j(eta)
                            double basis_xi  = amr::Basis::lagrange_1d(gl_pts, qi, xi);
                            double basis_eta = amr::Basis::lagrange_1d(gl_pts, qj, eta);
                            double basis_val = basis_xi * basis_eta;

                            const auto& qp_dof = cell_dof_tensor
                                [static_cast<unsigned int>(qi),
                                 static_cast<unsigned int>(qj)];

                            for (unsigned int d = 0; d < NumDOFs; ++d)
                            {
                                corner_dofs[d] +=
                                    static_cast<double>(qp_dof[d]) * basis_val;
                            }
                        }
                    }
                    point_dofs.push_back(corner_dofs);
                }

                // Add cell connectivity (quad: 4 points)
                connectivity.push_back(pt_base + 0);
                connectivity.push_back(pt_base + 1);
                connectivity.push_back(pt_base + 2);
                connectivity.push_back(pt_base + 3);

                offsets.push_back(static_cast<int>(connectivity.size()));
                types.push_back(9); // VTK_QUAD
            }
        }

        write_vtu_file(filename, points, connectivity, offsets, types, point_dofs);
        ++file_counter;
    }

private:
    void write_vtu_file(
        const std::string&                        filename,
        const std::vector<std::array<double, 3>>& points,
        const std::vector<int>&                   connectivity,
        const std::vector<int>&                   offsets,
        const std::vector<int>&                   types,
        const std::vector<std::vector<double>>&   point_data
    ) const
    {
        std::ofstream file(filename);
        if (!file)
        {
            std::cerr << "[DG_PRINTER_SIMPLE] ERROR: Cannot open " << filename << "\n";
            return;
        }

        file << R"(<?xml version="1.0"?>)" << "\n";
        file
            << R"(<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">)"
            << "\n";
        file << R"(  <UnstructuredGrid>)" << "\n";
        file << "    <Piece NumberOfPoints=\"" << points.size() << "\" NumberOfCells=\""
             << offsets.size() << "\">\n";

        // Write points
        file << "      <Points>\n";
        file << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" "
                "format=\"ascii\">\n";
        for (const auto& p : points)
        {
            file << std::scientific << std::setprecision(16) << p[0] << " " << p[1] << " "
                 << p[2] << "\n";
        }
        file << "        </DataArray>\n";
        file << "      </Points>\n";

        // Write cells
        file << "      <Cells>\n";
        file << "        <DataArray type=\"UInt32\" Name=\"connectivity\" "
                "format=\"ascii\">\n";
        for (int idx : connectivity)
            file << idx << " ";
        file << "\n";
        file << "        </DataArray>\n";

        file << "        <DataArray type=\"UInt32\" Name=\"offsets\" format=\"ascii\">\n";
        for (int off : offsets)
            file << off << "\n";
        file << "        </DataArray>\n";

        file << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
        for (int t : types)
            file << t << "\n";
        file << "        </DataArray>\n";
        file << "      </Cells>\n";

        // Write point data (DOF values at each point)
        if (!point_data.empty() && !point_data[0].empty())
        {
            file << "      <PointData>\n";
            for (std::size_t d = 0; d < NumDOFs; ++d)
            {
                file << "        <DataArray type=\"Float64\" Name=\"dof_" << d
                     << "\" format=\"ascii\">\n";
                for (const auto& point : point_data)
                {
                    if (d < point.size())
                        file << std::scientific << std::setprecision(16) << point[d]
                             << "\n";
                }
                file << "        </DataArray>\n";
            }
            file << "      </PointData>\n";
        }

        file << "    </Piece>\n";
        file << "  </UnstructuredGrid>\n";
        file << "</VTKFile>\n";
        file.close();

        std::cout << "[DG_PRINTER_SIMPLE] Wrote " << filename << "\n";
    }

    std::string m_base_filename;
};

} // namespace ndt::print

#endif // AMR_INCLUDED_NDT_PRINT_DG_TREE_FIXED
