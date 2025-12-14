#ifndef AMR_INCLUDED_PHYSICS_SYSTEM
#define AMR_INCLUDED_PHYSICS_SYSTEM

#include "ndtree/ndconcepts.hpp"
#include <array>
#include <cmath>
#include <cstddef>

namespace amr::ndt::solver
{

template <concepts::PatchIndex Patch_Index, concepts::PatchLayout Patch_Layout, auto Domain_Sizes>
class physics_system
{
public:
    using patch_index_t       = Patch_Index;
    using patch_layout_t      = Patch_Layout;
    using physics_coord_t     = double;
    using size_type           = typename patch_index_t::size_type;
    using level_t             = typename patch_index_t::level_t;
    
    static constexpr size_type n_dimension = patch_index_t::rank();
    using physics_coord_arr_t = std::array<physics_coord_t, n_dimension>;

private:
    static constexpr size_type s_max_depth = patch_index_t::max_depth();
    static constexpr uint32_t s_max_morton_coord = 1u << s_max_depth;
    
    // Get cells per patch dimension from the data layout
    static constexpr auto s_cells_per_patch_dim = []() constexpr {
        auto sizes = patch_layout_t::data_layout_t::shape_t::sizes();
        physics_coord_arr_t result{};
        for (size_type i = 0; i < n_dimension; ++i) {
            result[i] = static_cast<physics_coord_t>(sizes[n_dimension - 1 - i]);
        }
        return result;
    }();

    // Domain sizes - now per dimension
    static constexpr auto s_physics_lengths = []() constexpr {
        physics_coord_arr_t result{};
        for (size_type i = 0; i < n_dimension; ++i) {
            result[i] = static_cast<physics_coord_t>(Domain_Sizes[i]);
        }
        return result;
    }();

public:
    // Default constructor
    constexpr physics_system() = default;

    // Get the physical size of a patch in each dimension
    [[nodiscard]]
    static constexpr auto patch_sizes(patch_index_t const& morton_id) -> physics_coord_arr_t
    {
        auto [coords, level] = patch_index_t::decode(morton_id.id());
        uint32_t patch_size_morton = 1u << (s_max_depth - level);
        
        physics_coord_arr_t result{};
        for (size_type i = 0; i < n_dimension; ++i) {
            result[i] = s_physics_lengths[i] * static_cast<physics_coord_t>(patch_size_morton) 
                       / static_cast<physics_coord_t>(s_max_morton_coord);
        }
        return result;
    }

    // Get the physical size of a cell in each dimension
    [[nodiscard]]
    static constexpr auto cell_sizes(patch_index_t const& morton_id) -> physics_coord_arr_t
    {
        auto patch_size = patch_sizes(morton_id);
        physics_coord_arr_t result{};
        for (size_type i = 0; i < n_dimension; ++i) {
            result[i] = patch_size[i] / s_cells_per_patch_dim[i];
        }
        return result;
    }

    [[nodiscard]]
    static constexpr auto patch_coord(patch_index_t const& morton_id) -> physics_coord_arr_t
    {
        auto [coords, level] = patch_index_t::decode(morton_id.id());
        
        physics_coord_arr_t result{};
        for (size_type i = 0; i < n_dimension; ++i) {
            result[i] = s_physics_lengths[i] * static_cast<physics_coord_t>(coords[i])
                       / static_cast<physics_coord_t>(s_max_morton_coord);
            assert(result[i] <= s_physics_lengths[i] && "coord needs to be smaller than domain length");
        }
        return result;
    }

    [[nodiscard]]
    static constexpr auto cell_coord(patch_index_t const& morton_id, typename patch_layout_t::padded_layout_t::index_t linear_idx) -> physics_coord_arr_t
    {
        physics_coord_arr_t patch_origin = patch_coord(morton_id);
        auto cell_size = cell_sizes(morton_id);

        // Convert linear index (in padded layout) to multi-index
        auto multi_idx = patch_layout_t::padded_layout_t::multi_index(linear_idx);
        
        // The halo width offset to get to the data region
        static constexpr auto halo_width = patch_layout_t::halo_width();

        physics_coord_arr_t result{};
        for (size_type i = 0; i < n_dimension; ++i) {
            auto data_cell_idx = static_cast<physics_coord_t>(multi_idx[n_dimension - 1 - i]) - static_cast<physics_coord_t>(halo_width);
            result[i] = patch_origin[i] + data_cell_idx * cell_size[i];
            assert(result[i] <= s_physics_lengths[i] && "coord needs to be smaller than domain length");
        }
        return result;
    }

    [[nodiscard]]
    static constexpr auto domain_lengths() -> physics_coord_arr_t const&
    {
        return s_physics_lengths;
    }

    [[nodiscard]]
    static constexpr auto domain_length(size_type dim) -> physics_coord_t
    {
        return s_physics_lengths[dim];
    }
};

} // namespace amr::ndt::solver

#endif // AMR_INCLUDED_PHYSICS_SYSTEM
