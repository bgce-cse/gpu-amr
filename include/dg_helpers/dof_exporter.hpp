#ifndef AMR_DOF_EXPORTER_HPP
#define AMR_DOF_EXPORTER_HPP

#include "containers/static_layout.hpp"
#include "containers/static_tensor.hpp"
#include "containers/static_vector.hpp"
#include "ndtree/patch_layout.hpp"
#include "ndtree/patch_utils.hpp"
#include <fstream>
#include <iomanip>
#include <map>
#include <string>
#include <type_traits>
#include <vector>

namespace amr::ndt::exporter
{
// Exports DOFs to JSON with proper cell structure for visualization.
// - Points: cell corners (4 for 2D, 8 for 3D) plus all Gauss-Legendre nodes
// - Connectivity: cells with corners only
// - Point data: DOF values at each GL node
template <typename PatchDOFContainer, typename PatchCenterContainer, typename BasisType>
void export_patch_dofs_json(
    const PatchDOFContainer&    patch_dofs,
    const PatchCenterContainer& patch_centers,
    double                      cell_size,
    const BasisType&            basis,
    const std::string&          filename
)
{
    using layout_t       = typename PatchDOFContainer::layout_t;
    using patch_layout_t = amr::ndt::patches::patch_layout<layout_t, 1>;

    const unsigned int Order = basis.quadpoints().flat_size();
    const unsigned int Dim   = basis.dimensions();

    std::vector<std::array<double, 3>>          all_points;
    std::vector<std::vector<unsigned int>>      cell_connectivity;
    std::vector<std::vector<double>>            all_point_dofs;
    std::map<std::vector<double>, unsigned int> corner_dedup;

    // Process all non-halo cells
    for (std::size_t linear_idx = 0; linear_idx < patch_layout_t::flat_size();
         ++linear_idx)
    {
        if (amr::ndt::utils::patches::is_halo_cell<patch_layout_t>(linear_idx)) continue;

        const auto& doft = patch_dofs[linear_idx];
        using multi_idx_t =
            typename std::remove_reference_t<decltype(doft)>::multi_index_t;

        double cell_center_x = patch_centers[linear_idx][0];
        double cell_center_y = (Dim > 1) ? patch_centers[linear_idx][1] : 0.0;
        double cell_center_z = (Dim > 2) ? patch_centers[linear_idx][2] : 0.0;

        double half_cell = cell_size * 0.5;

        // Define cell corners
        std::vector<std::array<double, 3>> corners;
        if (Dim == 2)
        {
            // 4 corners for 2D quad
            corners = {
                { cell_center_x - half_cell, cell_center_y - half_cell, 0.0 },
                { cell_center_x + half_cell, cell_center_y - half_cell, 0.0 },
                { cell_center_x + half_cell, cell_center_y + half_cell, 0.0 },
                { cell_center_x - half_cell, cell_center_y + half_cell, 0.0 }
            };
        }
        else if (Dim == 3)
        {
            // 8 corners for 3D hex
            corners = {
                { cell_center_x - half_cell,
                 cell_center_y - half_cell,
                 cell_center_z - half_cell },
                { cell_center_x + half_cell,
                 cell_center_y - half_cell,
                 cell_center_z - half_cell },
                { cell_center_x + half_cell,
                 cell_center_y + half_cell,
                 cell_center_z - half_cell },
                { cell_center_x - half_cell,
                 cell_center_y + half_cell,
                 cell_center_z - half_cell },
                { cell_center_x - half_cell,
                 cell_center_y - half_cell,
                 cell_center_z + half_cell },
                { cell_center_x + half_cell,
                 cell_center_y - half_cell,
                 cell_center_z + half_cell },
                { cell_center_x + half_cell,
                 cell_center_y + half_cell,
                 cell_center_z + half_cell },
                { cell_center_x - half_cell,
                 cell_center_y + half_cell,
                 cell_center_z + half_cell }
            };
        }

        // Add cell corners to points list
        std::vector<unsigned int> cell_pt_indices;
        for (const auto& corner : corners)
        {
            std::vector<double> corner_key = { corner[0], corner[1], corner[2] };
            unsigned int        pt_idx;

            if (corner_dedup.find(corner_key) != corner_dedup.end())
            {
                pt_idx = corner_dedup[corner_key];
            }
            else
            {
                pt_idx                   = static_cast<unsigned int>(all_points.size());
                corner_dedup[corner_key] = pt_idx;
                all_points.push_back(corner);
                // Corners get average DOF value from all GL nodes in cell
                all_point_dofs.push_back(std::vector<double>());
            }

            cell_pt_indices.push_back(pt_idx);
        }

        // Now add all GL nodes as additional points with their DOF values
        multi_idx_t                      midx{};
        std::vector<std::vector<double>> gl_dofs;

        do
        {
            // Compute GL node coordinate
            std::vector<double> coords;
            for (unsigned d = 0; d < Dim; ++d)
            {
                double ref = basis.quadpoints()[midx[d]];
                double lower;
                if (d == 0)
                    lower = cell_center_x - half_cell;
                else if (d == 1)
                    lower = cell_center_y - half_cell;
                else
                    lower = cell_center_z - half_cell;
                double coord = lower + ref * cell_size;
                coords.push_back(coord);
            }
            while (coords.size() < 3)
                coords.push_back(0.0);

            // Add GL node as a point
            all_points.push_back({ coords[0], coords[1], coords[2] });

            // Store DOF values for this GL node
            const auto&         dof_vector = doft[midx];
            std::vector<double> dof_comps;
            for (unsigned c = 0; c < dof_vector.elements(); ++c)
            {
                dof_comps.push_back(dof_vector[c]);
            }
            all_point_dofs.push_back(dof_comps);
            gl_dofs.push_back(dof_comps);

        } while (midx.increment());

        // Compute average DOF for corners from GL nodes
        for (unsigned int corner_idx = 0; corner_idx < cell_pt_indices.size();
             ++corner_idx)
        {
            unsigned int corner_pt_idx = cell_pt_indices[corner_idx];
            if (!gl_dofs.empty() && !gl_dofs[0].empty())
            {
                std::vector<double> avg_dofs(gl_dofs[0].size(), 0.0);
                for (const auto& gl_dof : gl_dofs)
                {
                    for (std::size_t d = 0; d < gl_dof.size(); ++d)
                    {
                        avg_dofs[d] += gl_dof[d];
                    }
                }
                for (auto& val : avg_dofs)
                    val /= static_cast<double>(gl_dofs.size());
                all_point_dofs[corner_pt_idx] = avg_dofs;
            }
        }

        cell_connectivity.push_back(cell_pt_indices);
    }

    // Write JSON with VTK-compatible structure
    std::ofstream ofs(filename);
    if (!ofs.is_open()) return;
    ofs << std::fixed << std::setprecision(16);

    ofs << "{\n";
    ofs << "  \"metadata\": {\n";
    ofs << "    \"type\": \"DG_patch_cells\",\n";
    ofs << "    \"dimension\": " << Dim << ",\n";
    ofs << "    \"order\": " << Order << ",\n";
    ofs << "    \"cell_size\": " << cell_size << ",\n";
    ofs << "    \"num_cells\": " << cell_connectivity.size() << ",\n";
    ofs << "    \"num_points\": " << all_points.size() << "\n";
    ofs << "  },\n";

    // Write points
    ofs << "  \"points\": [\n";
    for (std::size_t i = 0; i < all_points.size(); ++i)
    {
        if (i > 0) ofs << ",\n";
        ofs << "    [" << all_points[i][0] << ", " << all_points[i][1] << ", "
            << all_points[i][2] << "]";
    }
    ofs << "\n  ],\n";

    // Write connectivity (cell corners only)
    ofs << "  \"connectivity\": [\n";
    for (std::size_t i = 0; i < cell_connectivity.size(); ++i)
    {
        if (i > 0) ofs << ",\n";
        ofs << "    [";
        const auto& cell = cell_connectivity[i];
        for (std::size_t j = 0; j < cell.size(); ++j)
        {
            if (j > 0) ofs << ", ";
            ofs << cell[j];
        }
        ofs << "]";
    }
    ofs << "\n  ],\n";

    // Write point data (DOF components at all points, corners and GL nodes)
    ofs << "  \"point_data\": {\n";
    if (!all_point_dofs.empty() && !all_point_dofs[0].empty())
    {
        unsigned int num_comps = static_cast<unsigned int>(all_point_dofs[0].size());
        for (unsigned int comp = 0; comp < num_comps; ++comp)
        {
            if (comp > 0) ofs << ",\n";
            ofs << "    \"dof_component_" << comp << "\": [";
            for (std::size_t pt = 0; pt < all_point_dofs.size(); ++pt)
            {
                if (pt > 0) ofs << ", ";
                if (comp < static_cast<unsigned int>(all_point_dofs[pt].size()))
                {
                    ofs << all_point_dofs[pt][comp];
                }
                else
                {
                    ofs << "0.0";
                }
            }
            ofs << "]";
        }
    }
    ofs << "\n  }\n";
    ofs << "}\n";
    ofs.close();
}

} // namespace amr::ndt::exporter

#endif // AMR_DOF_EXPORTER_HPP
