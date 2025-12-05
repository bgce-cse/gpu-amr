#ifndef AMR_INCLUDED_NDT_PRINT_DG_TREE
#define AMR_INCLUDED_NDT_PRINT_DG_TREE

#include "dg_helpers/basis.hpp"
#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

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

    using patch_layout_t = PatchLayout;
    using index_t        = typename PatchLayout::index_t;
    using size_type      = typename PatchLayout::size_type;

    // Basis for quadrature positions in [0,1] and evaluation helper
    amr::Basis::Basis<Order, Dim> basis{ 0.0, 1.0 };

    mutable int file_counter = 0;

    template <typename S1Tag, typename S3Tag, typename S4Tag, typename TreeT>
    void print(const TreeT& tree) const
    {
        using amr::ndt::utils::patches::is_halo_cell;

        std::ostringstream fn;
        fn << "dg_tree_output_" << std::setw(6) << std::setfill('0') << file_counter
           << ".vtu";
        std::string filename = fn.str();

        std::cout << "[DG_PRINTER] Writing " << filename << " (patches=" << tree.size()
                  << ", Dim=" << Dim << ", Order=" << Order << ", DOFs=" << NumDOFs
                  << ")\n";

        std::vector<std::array<double, 3>> all_points;
        std::vector<std::vector<double>>   point_dofs;
        std::vector<int>                   connectivity;
        std::vector<int>                   offsets;
        std::vector<int>                   types;

        all_points.reserve(4096);
        point_dofs.reserve(4096);
        connectivity.reserve(8192);
        offsets.reserve(1024);
        types.reserve(1024);

        const auto& gl_pts = basis.quadpoints();

        // Debug: print quadrature points
        std::cout << "[DG_PRINTER] quadpoints: ";
        for (std::size_t i = 0; i < Order; ++i)
        {
            auto idx = static_cast<
                typename std::remove_reference_t<decltype(gl_pts)>::size_type>(i);
            std::cout << gl_pts[idx] << (i + 1 < Order ? ", " : "\n");
        }

        for (std::size_t p = 0; p < tree.size(); ++p)
        {
            const auto& s1_patch = tree.template get_patch<S1Tag>(p).data();
            const auto& s3_patch = tree.template get_patch<S3Tag>(p).data();
            const auto& s4_patch = tree.template get_patch<S4Tag>(p).data();

            const std::size_t flat = patch_layout_t::flat_size();
            std::cout << "[DG_PRINTER] patch " << p << " flat_size=" << flat << "\n";

            using small_idx_t = unsigned int;

            for (std::size_t li = 0; li < flat; ++li)
            {
                if (is_halo_cell<patch_layout_t>(static_cast<index_t>(li))) continue;

                std::array<double, Dim> center{};
                std::array<double, Dim> size{};
                for (std::size_t d = 0; d < Dim; ++d)
                {
                    center[d] =
                        static_cast<double>(s3_patch[li][static_cast<small_idx_t>(d)]);
                    size[d] =
                        static_cast<double>(s4_patch[li][static_cast<small_idx_t>(d)]);
                }

                double hx = size[0] * 0.5;
                double hy = (Dim > 1 ? size[1] * 0.5 : 0.0);
                double hz = (Dim > 2 ? size[2] * 0.5 : 0.0);

                std::array<std::array<double, 3>, 8> corners{};
                if constexpr (Dim == 2)
                {
                    corners[0] = { center[0] - hx, center[1] - hy, 0.0 };
                    corners[1] = { center[0] + hx, center[1] - hy, 0.0 };
                    corners[2] = { center[0] + hx, center[1] + hy, 0.0 };
                    corners[3] = { center[0] - hx, center[1] + hy, 0.0 };
                }
                else
                {
                    corners[0] = { center[0] - hx, center[1] - hy, center[2] - hz };
                    corners[1] = { center[0] + hx, center[1] - hy, center[2] - hz };
                    corners[2] = { center[0] + hx, center[1] + hy, center[2] - hz };
                    corners[3] = { center[0] - hx, center[1] + hy, center[2] - hz };
                    corners[4] = { center[0] - hx, center[1] - hy, center[2] + hz };
                    corners[5] = { center[0] + hx, center[1] - hy, center[2] + hz };
                    corners[6] = { center[0] + hx, center[1] + hy, center[2] + hz };
                    corners[7] = { center[0] - hx, center[1] + hy, center[2] + hz };
                }

                const auto& cell_coeffs = s1_patch[li];
                using gl_idx_t =
                    typename std::remove_reference_t<decltype(gl_pts)>::size_type;

                // Build cell with geometry corners + DOF sampling points
                if constexpr (Dim == 2)
                {
                    if constexpr (Order == 1)
                    {
                        // Order 1: Single point at center
                        std::array<double, 3> gp{ center[0], center[1], 0.0 };
                        all_points.push_back(gp);
                        connectivity.push_back(static_cast<int>(all_points.size()) - 1);

                        auto dof_vec = cell_coeffs
                            [static_cast<small_idx_t>(0), static_cast<small_idx_t>(0)];
                        std::vector<double> vec(NumDOFs);
                        for (std::size_t d = 0; d < NumDOFs; ++d)
                            vec[d] =
                                static_cast<double>(dof_vec[static_cast<small_idx_t>(d)]);
                        point_dofs.push_back(std::move(vec));

                        offsets.push_back(static_cast<int>(connectivity.size()));
                        types.push_back(1); // VTK_VERTEX
                    }
                    else if constexpr (Order == 2)
                    {
                        // Order 2: 4 corners + 4 DOF points = 8 total
                        // Add 4 corner points
                        std::vector<int> corner_pts;
                        for (std::size_t c = 0; c < 4; ++c)
                        {
                            all_points.push_back(corners[c]);
                            corner_pts.push_back(static_cast<int>(all_points.size()) - 1);
                            std::vector<double> zero_dof(NumDOFs, 0.0);
                            point_dofs.push_back(zero_dof);
                        }

                        // Add 4 DOF points (GL quadrature points)
                        std::vector<int> dof_pts;
                        for (std::size_t j = 0; j < Order; ++j)
                        {
                            for (std::size_t i = 0; i < Order; ++i)
                            {
                                std::array<double, 3> gp{
                                    center[0] + (gl_pts[static_cast<gl_idx_t>(i)] - 0.5) *
                                                    size[0],
                                    center[1] + (gl_pts[static_cast<gl_idx_t>(j)] - 0.5) *
                                                    size[1],
                                    0.0
                                };
                                all_points.push_back(gp);
                                dof_pts.push_back(
                                    static_cast<int>(all_points.size()) - 1
                                );

                                auto dof_vec = cell_coeffs
                                    [static_cast<small_idx_t>(i),
                                     static_cast<small_idx_t>(j)];
                                std::vector<double> vec(NumDOFs);
                                for (std::size_t d = 0; d < NumDOFs; ++d)
                                    vec[d] = static_cast<double>(
                                        dof_vec[static_cast<small_idx_t>(d)]
                                    );
                                point_dofs.push_back(std::move(vec));
                            }
                        }

                        // Build 8-point quad: 4 corners + 4 DOF points
                        connectivity.push_back(corner_pts[0]); // BL
                        connectivity.push_back(corner_pts[1]); // BR
                        connectivity.push_back(corner_pts[2]); // TR
                        connectivity.push_back(corner_pts[3]); // TL
                        connectivity.push_back(dof_pts[0]);    // dof[0,0]
                        connectivity.push_back(dof_pts[1]);    // dof[1,0]
                        connectivity.push_back(dof_pts[3]);    // dof[0,1]
                        connectivity.push_back(dof_pts[2]);    // dof[1,1]

                        offsets.push_back(static_cast<int>(connectivity.size()));
                        types.push_back(9); // VTK_QUAD
                    }
                    else
                    {
                        // Order >= 3: All GL points
                        int cell_start = static_cast<int>(all_points.size());
                        for (std::size_t j = 0; j < Order; ++j)
                            for (std::size_t i = 0; i < Order; ++i)
                            {
                                std::array<double, 3> gp{
                                    center[0] + (gl_pts[static_cast<gl_idx_t>(i)] - 0.5) *
                                                    size[0],
                                    center[1] + (gl_pts[static_cast<gl_idx_t>(j)] - 0.5) *
                                                    size[1],
                                    0.0
                                };
                                all_points.push_back(gp);
                                connectivity.push_back(cell_start + (j * Order + i));

                                auto dof_vec = cell_coeffs
                                    [static_cast<small_idx_t>(i),
                                     static_cast<small_idx_t>(j)];
                                std::vector<double> vec(NumDOFs);
                                for (std::size_t d = 0; d < NumDOFs; ++d)
                                    vec[d] = static_cast<double>(
                                        dof_vec[static_cast<small_idx_t>(d)]
                                    );
                                point_dofs.push_back(std::move(vec));
                            }

                        offsets.push_back(static_cast<int>(connectivity.size()));
                        types.push_back(7); // VTK_POLYGON
                    }
                }
                else if constexpr (Dim == 3)
                {
                    if constexpr (Order == 1)
                    {
                        // Order 1: Single point at center
                        std::array<double, 3> gp{ center[0], center[1], center[2] };
                        all_points.push_back(gp);
                        connectivity.push_back(static_cast<int>(all_points.size()) - 1);

                        auto dof_vec = cell_coeffs
                            [static_cast<small_idx_t>(0),
                             static_cast<small_idx_t>(0),
                             static_cast<small_idx_t>(0)];
                        std::vector<double> vec(NumDOFs);
                        for (std::size_t d = 0; d < NumDOFs; ++d)
                            vec[d] =
                                static_cast<double>(dof_vec[static_cast<small_idx_t>(d)]);
                        point_dofs.push_back(std::move(vec));

                        offsets.push_back(static_cast<int>(connectivity.size()));
                        types.push_back(1); // VTK_VERTEX
                    }
                    else if constexpr (Order == 2)
                    {
                        // Order 2: 8 corners + 8 DOF points = 16 total
                        // Add 8 corner points
                        std::vector<int> corner_pts;
                        for (std::size_t c = 0; c < 8; ++c)
                        {
                            all_points.push_back(corners[c]);
                            corner_pts.push_back(static_cast<int>(all_points.size()) - 1);
                            std::vector<double> zero_dof(NumDOFs, 0.0);
                            point_dofs.push_back(zero_dof);
                        }

                        // Add 8 DOF points
                        std::vector<int> dof_pts;
                        for (std::size_t k = 0; k < Order; ++k)
                            for (std::size_t j = 0; j < Order; ++j)
                                for (std::size_t i = 0; i < Order; ++i)
                                {
                                    std::array<double, 3> gp{
                                        center[0] +
                                            (gl_pts[static_cast<gl_idx_t>(i)] - 0.5) *
                                                size[0],
                                        center[1] +
                                            (gl_pts[static_cast<gl_idx_t>(j)] - 0.5) *
                                                size[1],
                                        center[2] +
                                            (gl_pts[static_cast<gl_idx_t>(k)] - 0.5) *
                                                size[2]
                                    };
                                    all_points.push_back(gp);
                                    dof_pts.push_back(
                                        static_cast<int>(all_points.size()) - 1
                                    );

                                    auto dof_vec = cell_coeffs
                                        [static_cast<small_idx_t>(i),
                                         static_cast<small_idx_t>(j),
                                         static_cast<small_idx_t>(k)];
                                    std::vector<double> vec(NumDOFs);
                                    for (std::size_t d = 0; d < NumDOFs; ++d)
                                        vec[d] = static_cast<double>(
                                            dof_vec[static_cast<small_idx_t>(d)]
                                        );
                                    point_dofs.push_back(std::move(vec));
                                }

                        // Add all points to connectivity
                        for (const auto& idx : corner_pts)
                            connectivity.push_back(idx);
                        for (const auto& idx : dof_pts)
                            connectivity.push_back(idx);

                        offsets.push_back(static_cast<int>(connectivity.size()));
                        types.push_back(12); // VTK_HEXAHEDRON
                    }
                    else
                    {
                        // Order >= 3: All GL points
                        int cell_start = static_cast<int>(all_points.size());
                        for (std::size_t k = 0; k < Order; ++k)
                            for (std::size_t j = 0; j < Order; ++j)
                                for (std::size_t i = 0; i < Order; ++i)
                                {
                                    std::array<double, 3> gp{
                                        center[0] +
                                            (gl_pts[static_cast<gl_idx_t>(i)] - 0.5) *
                                                size[0],
                                        center[1] +
                                            (gl_pts[static_cast<gl_idx_t>(j)] - 0.5) *
                                                size[1],
                                        center[2] +
                                            (gl_pts[static_cast<gl_idx_t>(k)] - 0.5) *
                                                size[2]
                                    };
                                    all_points.push_back(gp);
                                    connectivity.push_back(
                                        cell_start + (k * Order * Order + j * Order + i)
                                    );

                                    auto dof_vec = cell_coeffs
                                        [static_cast<small_idx_t>(i),
                                         static_cast<small_idx_t>(j),
                                         static_cast<small_idx_t>(k)];
                                    std::vector<double> vec(NumDOFs);
                                    for (std::size_t d = 0; d < NumDOFs; ++d)
                                        vec[d] = static_cast<double>(
                                            dof_vec[static_cast<small_idx_t>(d)]
                                        );
                                    point_dofs.push_back(std::move(vec));
                                }

                        offsets.push_back(static_cast<int>(connectivity.size()));
                        types.push_back(7); // VTK_POLYGON
                    }
                }
            }
        }

        std::ofstream os(filename, std::ios::out);
        if (!os)
        {
            std::cerr << "[DG_PRINTER] ERROR: cannot open " << filename << "\n";
            return;
        }
        os << std::scientific << std::setprecision(12);

        os << "<?xml version=\"1.0\"?>\n";
        os << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" "
              "byte_order=\"LittleEndian\">\n";
        os << "  <UnstructuredGrid>\n";
        os << "    <Piece NumberOfPoints=\"" << all_points.size() << "\" NumberOfCells=\""
           << types.size() << "\">\n";

        os << "      <Points>\n";
        os << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" "
              "format=\"ascii\">\n";
        for (auto const& p : all_points)
            os << "          " << p[0] << " " << p[1] << " " << p[2] << "\n";
        os << "        </DataArray>\n";
        os << "      </Points>\n";

        os << "      <Cells>\n";
        os << "        <DataArray type=\"Int32\" Name=\"connectivity\" "
              "format=\"ascii\">\n";
        for (auto v : connectivity)
            os << "          " << v << "\n";
        os << "        </DataArray>\n";

        os << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
        for (auto v : offsets)
            os << "          " << v << "\n";
        os << "        </DataArray>\n";

        os << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
        for (auto v : types)
            os << "          " << v << "\n";
        os << "        </DataArray>\n";
        os << "      </Cells>\n";

        os << "      <PointData>\n";
        os << "        <DataArray type=\"Float64\" Name=\"DG_quad_DOFS\" "
              "NumberOfComponents=\""
           << NumDOFs << "\" format=\"ascii\">\n";
        for (auto const& vec : point_dofs)
        {
            os << "          ";
            for (std::size_t k = 0; k < vec.size(); ++k)
                os << vec[k] << (k + 1 < vec.size() ? " " : "");
            os << "\n";
        }
        os << "        </DataArray>\n";
        os << "      </PointData>\n";

        os << "    </Piece>\n";
        os << "  </UnstructuredGrid>\n";
        os << "</VTKFile>\n";

        os.close();
        ++file_counter;
        std::cout << "[DG_PRINTER] Wrote " << filename << "\n";
    }
};

} // namespace ndt::print

#endif // AMR_INCLUDED_NDT_PRINT_DG_TREE
