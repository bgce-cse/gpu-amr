#pragma once

#include "data_types.hpp"
#include "domain.hpp"
#include "tree_types.hpp"
// #include "morton_id.hpp"
// #include "tree.hpp"
#include <algorithm>
#include <bitset>
#include <filesystem>
#include <fstream>
#include <map>
#include <ostream>
#include <ranges>

class vtk_printer
{
  public:
    vtk_printer() = default;
    vtk_printer(std::string base_filename)
        : m_base_filename(std::move(base_filename))
    {
    }

    void print(sim_domain& domain, int timestep) const
    {
        std::string filename_extension = std::to_string(timestep) + ".vtk";
        std::string full_filename =
            m_base_filename + "/result" + filename_extension;
        std::filesystem::path full_path(full_filename);
        std::filesystem::create_directories(full_path.parent_path());

        std::ofstream file(full_filename);
        if (!file.is_open())
        {
            throw std::runtime_error("Cannot open file: " + full_filename);
        }
        write_header(file);
        write_points(file, domain);
        file.flush();
    }

  private:
    void write_header(std::ofstream& file) const
    {
        file << "# vtk DataFile Version 3.0\n";
        file << "AMR Tree Structure\n";
        file << "ASCII\n";
        file << "DATASET UNSTRUCTURED_GRID\n";
    }

    void write_points(std::ofstream& file, sim_domain& domain) const
    {

        std::vector<std::array<double, 3>> points;
        std::vector<size_t> cell_indices;
        auto max_depth = index_t::max_depth();
        double max_coordinate = 1u << max_depth;

        std::map<std::array<double, 2>, std::vector<index_t>> vertex_to_cells;

        for (size_t i = 0; i < domain.tree().size(); ++i)
        {
            auto id = domain.tree().get_node_index_at(i);
            auto level = id.level();
            double cell_size = 1u << (max_depth - level);
            double cell_size_x = cell_size * domain.x_length();
            double cell_size_y = cell_size * domain.y_length();
            auto [coords, _] = index_t::decode(id);
            double x = static_cast<double>(coords[0]) * domain.x_length();
            double y =
                (max_coordinate - static_cast<double>(coords[1]) - cell_size) *
                domain.y_length();

            // Map each corner to the owning cell
            vertex_to_cells[{x, y}].push_back(id);
            vertex_to_cells[{x + cell_size_x, y}].push_back(id);
            vertex_to_cells[{x + cell_size_x, y + cell_size_y}].push_back(id);
            vertex_to_cells[{x, y + cell_size_y}].push_back(id);

            cell_indices.push_back(points.size());
            points.push_back({x, y, 0});
            points.push_back({x + cell_size_x, y, 0});
            points.push_back({x + cell_size_x, y + cell_size_y, 0});
            points.push_back({x, y + cell_size_y, 0});
        }

        // Write points
        file << "POINTS " << points.size() << " double\n";
        for (auto const& [x, y, z] : points)
        {
            file << x << " " << y << " " << z << "\n";
        }

        // Write cells
        file << "CELLS " << cell_indices.size() << " "
             << cell_indices.size() * 5 << "\n";
        for (size_t i = 0; i < cell_indices.size(); ++i)
        {
            size_t idx = cell_indices[i];
            file << "4 " << idx << " " << idx + 1 << " " << idx + 2 << " "
                 << idx + 3 << "\n";
        }

        // Cell types (VTK_QUAD = 9)
        file << "CELL_TYPES " << cell_indices.size() << "\n";
        for (size_t i = 0; i < cell_indices.size(); ++i)
        {
            file << "9\n";
        }

        // --- CELL DATA ---
        file << "CELL_DATA " << cell_indices.size() << "\n";

        file << "SCALARS cell_type int 1\nLOOKUP_TABLE default\n";
        for (size_t i = 0; i < cell_indices.size(); ++i)
        {
            auto id = domain.tree().get_node_index_at(i);
            auto cell_type = domain.tree().get_cell_type(id);
            int type = cell_type == cell_type::FLUID ? 1 : 0;
            file << type << "\n";
        }

        // Pressure
        file << "SCALARS p float 1\nLOOKUP_TABLE default\n";
        for (size_t i = 0; i < cell_indices.size(); ++i)
        {
            auto id = domain.tree().get_node_index_at(i);
            file << domain.tree().template get<p_t>(id) << "\n";
        }

        // Temperature
        file << "SCALARS t float 1\nLOOKUP_TABLE default\n";
        for (size_t i = 0; i < cell_indices.size(); ++i)
        {
            auto id = domain.tree().get_node_index_at(i);
            file << domain.tree().template get<t_t>(id) << "\n";
        }

        // Velocity components
        file << "SCALARS u_velo float 1\nLOOKUP_TABLE default\n";
        for (size_t i = 0; i < cell_indices.size(); ++i)
        {
            auto id = domain.tree().get_node_index_at(i);
            file << domain.tree().template get<u_t>(id) << "\n";
        }

        file << "SCALARS v_velo float 1\nLOOKUP_TABLE default\n";
        for (size_t i = 0; i < cell_indices.size(); ++i)
        {
            auto id = domain.tree().get_node_index_at(i);
            file << domain.tree().template get<v_t>(id) << "\n";
        }

        file << "SCALARS f_flux float 1\nLOOKUP_TABLE default\n";
        for (size_t i = 0; i < cell_indices.size(); ++i)
        {
            auto id = domain.tree().get_node_index_at(i);
            file << domain.tree().template get<f_t>(id) << "\n";
        }

        file << "SCALARS g_flux float 1\nLOOKUP_TABLE default\n";
        for (size_t i = 0; i < cell_indices.size(); ++i)
        {
            auto id = domain.tree().get_node_index_at(i);
            file << domain.tree().template get<g_t>(id) << "\n";
        }
        file << "SCALARS rhs float 1\nLOOKUP_TABLE default\n";
        for (size_t i = 0; i < cell_indices.size(); ++i)
        {
            auto id = domain.tree().get_node_index_at(i);
            file << domain.tree().template get<rhs_t>(id) << "\n";
        }

        file << "SCALARS total_velo float 1\nLOOKUP_TABLE default\n";
        for (size_t i = 0; i < cell_indices.size(); ++i)
        {
            auto id = domain.tree().get_node_index_at(i);
            auto u = domain.tree().template get<u_t>(id);
            auto v = domain.tree().template get<v_t>(id);
            file << std::sqrt(u * u + v * v) << "\n";
        }

        // --- POINT DATA (interpolated velocity vectors) ---
        file << "POINT_DATA " << points.size() << "\n";
        file << "VECTORS velocity float\n";

        for (const auto& [x, y, z] : points)
        {
            std::array<double, 2> key = {x, y};
            const auto& owner_cells = vertex_to_cells[key];

            double u_avg = 0.0, v_avg = 0.0;
            for (const auto& id : owner_cells)
            {
                u_avg += domain.tree().template get<u_t>(id);
                v_avg += domain.tree().template get<v_t>(id);
            }

            if (!owner_cells.empty())
            {
                u_avg /= static_cast<double>(owner_cells.size());
                v_avg /= static_cast<double>(owner_cells.size());
            }

            file << u_avg << " " << v_avg << " 0.0\n";
        }
    }

    std::string m_base_filename;
};

struct structured_print
{
  public:
    structured_print(std::ostream& os) noexcept
        : m_os(os)
    {
    }

    template <typename Tree>
    auto print(Tree const& tree) const -> void
    {
        assert(tree.is_sorted());
        for (auto i = 0uz; i != tree.size(); ++i)
        {
            const auto idx = tree.get_node_index_at(i);
            const auto [coords, level] = index_t::decode(idx);
            print_header(m_os, index_t::level(idx))
                << "h: " << std::bitset<index_t::bits()>(idx.id()).to_string()
                << ", offset: " << index_t::offset_of(idx)
                << ", x: " << coords[0] << " ,y: " << coords[1]
                << ", level: " << +level << ", idx: " << i << '\n';
        }
    }

  private:
    static auto print_header(std::ostream& os, auto depth) -> std::ostream&
    {
        for (auto i = decltype(depth){}; i != depth; ++i)
        {
            os << '\t';
        }
        return os;
    }

    std::ostream& m_os;
};
