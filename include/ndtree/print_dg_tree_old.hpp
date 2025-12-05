#ifndef AMR_INCLUDED_NDT_PRINT_DG_TREE
#define AMR_INCLUDED_NDT_PRINT_DG_TREE

#include "dg_helpers/basis.hpp"
#include <iostream>
#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkDoubleArray.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkQuad.h>
#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLUnstructuredGridWriter.h>

namespace ndt::print
{

template <
    std::size_t Dim,
    std::size_t Order,
    std::size_t PatchSize,
    std::size_t HaloWidth,
    std::size_t NumDOFs,
    typename PatchLayout>
struct dg_tree_printer
{
    static constexpr std::size_t dim        = Dim;
    static constexpr std::size_t order      = Order;
    static constexpr std::size_t patch_size = PatchSize;
    static constexpr std::size_t halo       = HaloWidth;
    static constexpr std::size_t ndofs      = NumDOFs;

    using patch_layout_t  = PatchLayout;
    using padded_layout_t = typename PatchLayout::padded_layout_t;
    using index_t         = typename PatchLayout::index_t;
    using size_type       = typename PatchLayout::size_type;
    using multi_index_t   = typename padded_layout_t::multi_index_t;

    amr::Basis::GaussLegendre<Order> gl_helper{ 0.0, 1.0 };

    mutable int file_counter = 0;

    // Placeholder function for vertex DOF values
    double vertex_value(const std::array<double, 3>& /*vertex*/) const
    {
        return 0.0; // replace with user-defined function
    }

    template <typename S1Tag, typename S3Tag, typename S4Tag>
    void print(const auto& tree) const
    {
        using namespace amr::ndt::utils::patches;

        vtkSmartPointer<vtkPoints>      points = vtkSmartPointer<vtkPoints>::New();
        vtkSmartPointer<vtkDoubleArray> values = vtkSmartPointer<vtkDoubleArray>::New();
        values->SetNumberOfComponents(NumDOFs);
        values->SetName("DG_DOFs");

        vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();

        vtkSmartPointer<vtkUnstructuredGrid> grid =
            vtkSmartPointer<vtkUnstructuredGrid>::New();

        const auto& gl_points = gl_helper.points();

        std::cout << "[INFO] Starting DG tree print...\n";
        std::cout << "[DEBUG] Gauss-Legendre points collection initialized\n";

        vtkIdType   point_id          = 0;
        std::size_t total_cells       = 0;
        std::size_t total_quad_points = 0;

        for (std::size_t patch_idx = 0; patch_idx < tree.size(); ++patch_idx)
        {
            const auto& dof_patch     = tree.template get_patch<S1Tag>(patch_idx);
            const auto& centers_patch = tree.template get_patch<S3Tag>(patch_idx);
            const auto& size_patch    = tree.template get_patch<S4Tag>(patch_idx);

            for (size_type idx = 0; idx < patch_layout_t::flat_size(); ++idx)
            {
                if (is_halo_cell<patch_layout_t>(idx)) continue;

                const auto& cell_center = centers_patch[idx];
                const auto& cell_size   = size_patch[idx];
                const auto& cell_dofs   = dof_patch[idx];

                std::cout << "[DEBUG] Processing cell at center (" << cell_center[0]
                          << ", " << cell_center[1] << ")\n";

                // -------------------------
                // 1. Insert corner vertices with placeholder DOF values
                // -------------------------
                std::array<std::array<double, 3>, 4> corners = {
                    { { cell_center[0] - 0.5 * cell_size[0],
                        cell_center[1] - 0.5 * cell_size[1],
                        0.0 },
                     { cell_center[0] + 0.5 * cell_size[0],
                        cell_center[1] - 0.5 * cell_size[1],
                        0.0 },
                     { cell_center[0] + 0.5 * cell_size[0],
                        cell_center[1] + 0.5 * cell_size[1],
                        0.0 },
                     { cell_center[0] - 0.5 * cell_size[0],
                        cell_center[1] + 0.5 * cell_size[1],
                        0.0 } }
                };

                vtkIdList* cell_point_ids = vtkIdList::New();

                for (int i = 0; i < 4; ++i)
                {
                    double pid = points->InsertNextPoint(
                        corners[i][0], corners[i][1], corners[i][2]
                    );

                    // Placeholder DOF values for corner vertices
                    std::array<double, NumDOFs> corner_dofs{};
                    for (unsigned int d = 0; d < NumDOFs; ++d)
                    {
                        corner_dofs[d] = 0.0; // placeholder
                    }
                    values->InsertNextTuple(corner_dofs.data());
                    cell_point_ids->InsertNextId(pid);
                }

                cells->InsertNextCell(4, cell_point_ids->GetPointer(0));
                cell_point_ids->Delete();

                // -------------------------
                // 2. Insert quadrature points with DOF values from patch
                // -------------------------
                auto multi_idx = multi_index_t();

                [[maybe_unused]]
                std::size_t quad_count = 0;

                do
                {
                    // Compute position in physical space
                    std::array<double, 3> pt = { 0.0, 0.0, 0.0 };

                    if constexpr (Dim >= 1)
                        pt[0] =
                            cell_center[0] +
                            (gl_points[static_cast<unsigned int>(multi_idx[0])] - 0.5) *
                                cell_size[0];

                    if constexpr (Dim >= 2)
                        pt[1] =
                            cell_center[1] +
                            (gl_points[static_cast<unsigned int>(multi_idx[1])] - 0.5) *
                                cell_size[1];

                    if constexpr (Dim >= 3)
                        pt[2] =
                            cell_center[2] +
                            (gl_points[static_cast<unsigned int>(multi_idx[2])] - 0.5) *
                                cell_size[2];

                    [[maybe_unused]]
                    double pid = points->InsertNextPoint(pt[0], pt[1], pt[2]);

                    // Get DOF values at this quadrature node from the patch
                    // cell_dofs is a static_tensor<static_vector<double, NumDOFs>, ...>
                    // Use if constexpr to avoid C++23 comma-subscript issues
                    const auto& dof_values = [&]()
                    {
                        if constexpr (Dim == 2)
                        {
                            return cell_dofs
                                [static_cast<std::size_t>(multi_idx[0]),
                                 static_cast<std::size_t>(multi_idx[1])];
                        }
                        else
                        {
                            return cell_dofs
                                [static_cast<std::size_t>(multi_idx[0]),
                                 static_cast<std::size_t>(multi_idx[1]),
                                 static_cast<std::size_t>(multi_idx[2])];
                        }
                    }();

                    // Create tuple for this point - dof_values is static_vector<double,
                    // NumDOFs>
                    std::array<double, NumDOFs> dof_tuple{};
                    for (unsigned int d = 0; d < NumDOFs; ++d)
                    {
                        dof_tuple[d] = dof_values[d];
                    }
                    values->InsertNextTuple(dof_tuple.data());
                    quad_count++;
                    total_quad_points++;

                } while (multi_idx.increment());

                total_cells++;
            }
        }

        // Finalize grid
        grid->SetPoints(points);
        grid->SetCells(VTK_QUAD, cells);
        grid->GetPointData()->SetScalars(values);

        std::cout << "[INFO] Total cells: " << total_cells << "\n";
        std::cout << "[INFO] Total quadrature points: " << total_quad_points << "\n";
        std::cout << "[INFO] Total points in grid: " << points->GetNumberOfPoints()
                  << "\n";

        // Write VTK file
        vtkSmartPointer<vtkXMLUnstructuredGridWriter> writer =
            vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
        std::string filename = "dg_output_" + std::to_string(file_counter) + ".vtu";
        writer->SetFileName(filename.c_str());
        writer->SetInputData(grid);
        try
        {
            writer->Write();
            std::cout << "[INFO] Wrote VTK file: " << filename << "\n";
        }
        catch (const std::exception& e)
        {
            std::cerr << "[ERROR] Failed to write VTK file: " << e.what() << "\n";
        }

        mutable_cast(file_counter)++;
    }

    // -------------------------
    // 2. Insert quadrature points with actual DOF values from patch
    // -------------------------
    multi_index_t multi_idx{};
    int           quad_count = 0;
    do
    {
        // Compute quadrature point position
        double pt[3] = { 0, 0, 0 };
        pt[0]        = cell_center[0] +
                (gl_points[static_cast<unsigned int>(multi_idx[0])] - 0.5) * cell_size[0];
        if (Dim >= 2)
            pt[1] =
                cell_center[1] +
                (gl_points[static_cast<unsigned int>(multi_idx[1])] - 0.5) * cell_size[1];
        if (Dim >= 3)
            pt[2] =
                cell_center[2] +
                (gl_points[static_cast<unsigned int>(multi_idx[2])] - 0.5) * cell_size[2];

        [[maybe_unused]]
        vtkIdType pid = points->InsertNextPoint(pt[0], pt[1], pt[2]);

        // Get DOF values at this quadrature node from the patch
        // cell_dofs is a static_tensor<static_vector<double, NumDOFs>, ...>
        // Use if constexpr to avoid C++23 comma-subscript issues
        const auto& dof_values = [&]()
        {
            if constexpr (Dim == 2)
            {
                return cell_dofs
                    [static_cast<std::size_t>(multi_idx[0]),
                     static_cast<std::size_t>(multi_idx[1])];
            }
            else
            {
                return cell_dofs
                    [static_cast<std::size_t>(multi_idx[0]),
                     static_cast<std::size_t>(multi_idx[1]),
                     static_cast<std::size_t>(multi_idx[2])];
            }
        }();

        // Create tuple for this point - dof_values is static_vector<double,
        // NumDOFs>
        std::array<double, NumDOFs> dof_tuple{};
        for (unsigned int d = 0; d < NumDOFs; ++d)
        {
            dof_tuple[d] = dof_values[d];
        }
        values->InsertNextTuple(dof_tuple.data());
        quad_count++;

    } while (multi_idx.increment());

    std::cout << "[DEBUG] Cell has " << quad_count << " quadrature points\n";
}
}

grid->SetPoints(points);
grid->GetPointData()->AddArray(values);

std::string name = "dg_tree_output_" + std::to_string(file_counter++) + ".vtu";
vtkSmartPointer<vtkXMLUnstructuredGridWriter> writer =
    vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
writer->SetFileName(name.c_str());
writer->SetInputData(grid);
writer->Write();

std::cout << "[INFO] Finished writing VTK file: " << name << "\n";
}
}
;

} // namespace ndt::print

#endif
