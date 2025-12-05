#ifndef AMR_INCLUDED_NDT_PRINT_DG_TREE_2D
#define AMR_INCLUDED_NDT_PRINT_DG_TREE_2D

#include "dg_helpers/basis.hpp"
#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>

namespace ndt::print
{

template <
    std::size_t Order,
    std::size_t PatchSize,
    std::size_t HaloWidth,
    std::size_t NumDOFs,
    typename PatchLayout>
struct dg_tree_printer_2d
{
    static constexpr std::size_t padded_size = PatchSize + 2 * HaloWidth;

    amr::Basis::Basis<Order, 2>                         basis{ 0.0, 1.0 };
    mutable int                                         file_counter = 0;
    mutable std::vector<std::pair<std::string, double>> output_files;

    template <typename S1Tag, typename TreeT>
    void print(const TreeT& tree, double time = 0.0) const
    {
        std::ostringstream fn;
        fn << "vtk_output/dg_tree_Order" << Order << "_" << std::setw(6)
           << std::setfill('0') << file_counter << ".vtu";
        std::string filename = fn.str();

        std::cout << "[PRINTER_2D] Writing " << filename << "\n";

        std::vector<std::array<double, 3>>                                points;
        std::vector<std::vector<unsigned int>>                            cells;
        std::vector<std::vector<double>>                                  point_data;
        std::map<std::tuple<std::size_t, std::size_t, int>, unsigned int> point_map;

        const auto& gl_pts    = basis.quadpoints();
        double      cell_size = 0.1;

        for (std::size_t patch_idx = 0; patch_idx < tree.size(); ++patch_idx)
        {
            const auto& dof_patch = tree.template get_patch<S1Tag>(patch_idx);
            std::size_t flat_size = padded_size * padded_size;

            for (std::size_t li = 0; li < flat_size; ++li)
            {
                std::size_t local_y = li / padded_size;
                std::size_t local_x = li % padded_size;

                // Skip halo
                if (local_x < HaloWidth || local_x >= PatchSize + HaloWidth ||
                    local_y < HaloWidth || local_y >= PatchSize + HaloWidth)
                    continue;

                const auto& cell_dofs = dof_patch[li];

                std::size_t grid_x = patch_idx * PatchSize + (local_x - HaloWidth);
                std::size_t grid_y = local_y - HaloWidth;

                double center_x  = (static_cast<double>(grid_x) + 0.5) * cell_size;
                double center_y  = (static_cast<double>(grid_y) + 0.5) * cell_size;
                double half_cell = cell_size * 0.5;

                std::array<std::array<double, 3>, 4> corners = {
                    { { center_x - half_cell, center_y - half_cell, 0.0 },
                     { center_x + half_cell, center_y - half_cell, 0.0 },
                     { center_x + half_cell, center_y + half_cell, 0.0 },
                     { center_x - half_cell, center_y + half_cell, 0.0 } }
                };

                std::vector<unsigned int> cell_points;

                for (int corner = 0; corner < 4; ++corner)
                {
                    auto pt_key = std::make_tuple(grid_x, grid_y, corner);

                    unsigned int pt_idx;
                    auto         iter = point_map.find(pt_key);
                    if (iter != point_map.end())
                    {
                        pt_idx = iter->second;
                    }
                    else
                    {
                        pt_idx            = static_cast<unsigned int>(points.size());
                        point_map[pt_key] = pt_idx;
                        points.push_back(corners[corner]);

                        // Compute corner position in reference space [-1, 1]²
                        // corner 0: (-1, -1), corner 1: (1, -1)
                        // corner 2: (1, 1),   corner 3: (-1, 1)
                        double xi  = (corner == 0 || corner == 3) ? -1.0 : 1.0;
                        double eta = (corner == 0 || corner == 1) ? -1.0 : 1.0;

                        // Evaluate basis functions at corner position
                        std::vector<double> dof_vals(NumDOFs, 0.0);

                        using gl_idx_t =
                            typename std::remove_reference_t<decltype(gl_pts)>::size_type;

                        for (std::size_t qj = 0; qj < Order; ++qj)
                        {
                            for (std::size_t qi = 0; qi < Order; ++qi)
                            {
                                const auto& qp_dof = cell_dofs
                                    [static_cast<gl_idx_t>(qi),
                                     static_cast<gl_idx_t>(qj)];

                                // Evaluate 1D Lagrange basis at corner position
                                double basis_xi = amr::Basis::lagrange_1d(gl_pts, qi, xi);
                                double basis_eta =
                                    amr::Basis::lagrange_1d(gl_pts, qj, eta);
                                double basis_val = basis_xi * basis_eta;

                                for (unsigned int d = 0; d < NumDOFs; ++d)
                                {
                                    dof_vals[d] +=
                                        static_cast<double>(qp_dof[d]) * basis_val;
                                }
                            }
                        }

                        point_data.push_back(dof_vals);
                    }

                    cell_points.push_back(pt_idx);
                }

                cells.push_back(cell_points);
            }
        }

        write_vtu(filename, points, cells, point_data);
        output_files.emplace_back(filename, time);
        ++file_counter;
    }

private:
    void write_pvd(const std::string& pvd_filename) const
    {
        std::ofstream file(pvd_filename);
        file << R"(<?xml version="1.0"?>)" << "\n";
        file << R"(<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">)"
             << "\n";
        file << R"(  <Collection>)" << "\n";

        for (const auto& [vtu_file, time] : output_files)
        {
            file << "    <DataSet timestep=\"" << time
                 << "\" group=\"\" part=\"0\" file=\"" << vtu_file << "\"/>\n";
        }

        file << R"(  </Collection>)" << "\n";
        file << R"(</VTKFile>)" << "\n";
        file.close();
        std::cout << "[PRINTER_2D] Generated PVD file: " << pvd_filename << "\n";
    }

public:
    void generate_pvd(const std::string& pvd_filename) const
    {
        write_pvd(pvd_filename);
    }

private:
    void write_vtu(
        const std::string&                            filename,
        const std::vector<std::array<double, 3>>&     points,
        const std::vector<std::vector<unsigned int>>& cells,
        const std::vector<std::vector<double>>&       point_data
    ) const
    {
        std::ofstream file(filename);
        file << R"(<?xml version="1.0"?>)" << "\n";
        file
            << R"(<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">)"
            << "\n";
        file << R"(  <UnstructuredGrid>)" << "\n";
        file << "    <Piece NumberOfPoints=\"" << points.size() << "\" NumberOfCells=\""
             << cells.size() << "\">\n";

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

        file << "      <Cells>\n";
        file << "        <DataArray type=\"UInt32\" Name=\"connectivity\" "
                "format=\"ascii\">\n";
        for (const auto& c : cells)
        {
            for (unsigned int idx : c)
                file << idx << " ";
            file << "\n";
        }
        file << "        </DataArray>\n";

        file << "        <DataArray type=\"UInt32\" Name=\"offsets\" format=\"ascii\">\n";
        unsigned int off = 0;
        for (const auto& c : cells)
        {
            off += static_cast<unsigned int>(c.size());
            file << off << "\n";
        }
        file << "        </DataArray>\n";

        file << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
        for (std::size_t i = 0; i < cells.size(); ++i)
            file << "9\n";
        file << "        </DataArray>\n";
        file << "      </Cells>\n";

        if (!point_data.empty())
        {
            file << "      <PointData>\n";
            for (std::size_t d = 0; d < NumDOFs; ++d)
            {
                file << "        <DataArray type=\"Float64\" Name=\"dof_" << d
                     << "\" format=\"ascii\">\n";
                for (const auto& pd : point_data)
                {
                    if (d < pd.size())
                        file << std::scientific << std::setprecision(16) << pd[d] << "\n";
                }
                file << "        </DataArray>\n";
            }
            file << "      </PointData>\n";
        }

        file << "    </Piece>\n";
        file << "  </UnstructuredGrid>\n";
        file << "</VTKFile>\n";
        file.close();

        std::cout << "[PRINTER_2D] Wrote " << filename << "\n";
    }
};

} // namespace ndt::print

#endif
