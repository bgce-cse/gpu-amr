#ifndef DG_PATCHES_HPP
#define DG_PATCHES_HPP

#include "../build/generated_config.hpp"
#include "containers/container_utils.hpp"
#include "containers/static_layout.hpp"
#include "containers/static_shape.hpp"
#include "containers/static_vector.hpp"
#include "dg_helpers/basis/basis.hpp"
#include "dg_helpers/equations/equations.hpp"
#include "dg_helpers/globals/global_config.hpp"
#include "dg_helpers/globals/globals.hpp"
#include "dg_helpers/rhs.hpp"
#include "dg_helpers/surface.hpp"
#include "dg_helpers/time_integration/time_integration.hpp"
#include "morton/morton_id.hpp"
#include "ndtree/ndhierarchy.hpp"
#include "ndtree/ndtree.hpp"
#include "ndtree/patch_layout.hpp"

namespace amr::dg_tree
{
template <typename global_t, typename Policy>
struct TreeBuilder
{
    struct S1
    {
        using dof_value_t = amr::containers::static_vector<double, Policy::DOFs>;
        using type        = typename amr::containers::utils::types::tensor::
            hypercube_t<dof_value_t, Policy::Order, Policy::Dim>;

        static constexpr auto index() noexcept -> std::size_t
        {
            return 0;
        }

        type value;
    };

    struct S2
    {
        using dof_value_t = amr::containers::static_vector<double, Policy::DOFs>;
        using dof_t       = typename amr::containers::utils::types::tensor::
            hypercube_t<dof_value_t, Policy::Order, Policy::Dim>;
        using type = amr::containers::static_vector<dof_t, Policy::Dim>;

        static constexpr auto index() noexcept -> std::size_t
        {
            return 1;
        }

        type value;
    };

    struct S3
    {
        using type = amr::containers::static_vector<double, Policy::Dim>;

        static constexpr auto index() noexcept -> std::size_t
        {
            return 2;
        }

        type value;
    };

    struct S4
    {
        using type = amr::containers::static_vector<double, Policy::Dim>;

        static constexpr auto index() noexcept -> std::size_t
        {
            return 3;
        }

        type value;
    }; // End of S4

    // Marker cell type for tree AMR with DG data (DOF tensor, flux, center)

    struct MarkerCell
    {
        using deconstructed_types_map_t = std::tuple<S1, S2, S3, S4>;

        MarkerCell() = default;

        auto data_tuple() -> auto&
        {
            return m_data;
        }

        auto data_tuple() const -> auto const&
        {
            return m_data;
        }

        deconstructed_types_map_t m_data;
    };

    using shape_t =
        amr::containers::static_shape<Policy::PatchSize, Policy::PatchSize>; // PatchSize
                                                                             // literal
    using layout_t      = amr::containers::static_layout<shape_t>;
    using patch_index_t = amr::ndt::morton::morton_id<
        Policy::MaxDepth,
        static_cast<unsigned int>(Policy::Dim)>; // Morton 2D with depth 1
    using patch_layout_t =
        amr::ndt::patches::patch_layout<layout_t, 1>; // HaloWidth literal
    using tree_t = amr::ndt::tree::ndtree<MarkerCell, patch_index_t, patch_layout_t>;

    tree_t tree{ 100000 };

    TreeBuilder()
    {
        for (std::size_t idx = 0; idx < tree.size(); ++idx)
        {
            auto& dof_patch          = tree.template get_patch<S1>(idx);
            auto& flux_patch         = tree.template get_patch<S2>(idx);
            auto& center_coord_patch = tree.template get_patch<S3>(idx);
            auto& size_patch         = tree.template get_patch<S4>(idx);

            auto patch_id                    = patch_index_t(idx);
            auto [patch_coords, patch_level] = patch_index_t::decode(patch_id.id());
            double patch_level_size = 1.0 / static_cast<double>(1u << patch_level);
            double cell_size = patch_level_size / static_cast<double>(Policy::PatchSize);

            for (typename patch_layout_t::index_t linear_idx = 0;
                 linear_idx != patch_layout_t::flat_size();
                 ++linear_idx)
            {
                if (amr::ndt::utils::patches::is_halo_cell<patch_layout_t>(linear_idx))
                    continue;

                auto coords_with_halo = global_t::lin_to_local(linear_idx);
                auto local_indices    = global_t::rm_halo(coords_with_halo);
                auto cell_center      = global_t::compute_center(patch_id, local_indices);

                center_coord_patch[linear_idx] = cell_center;

                dof_patch[linear_idx] =
                    global_t::interpolate_initial_dofs(cell_center, cell_size);

                flux_patch[linear_idx] =
                    global_t::EquationImpl::evaluate_flux(dof_patch[linear_idx]);

                size_patch[linear_idx] = { cell_size,
                                           cell_size }; // TODO generalize for ND
            }
        }
    };
};

} // namespace amr::dg_tree

#endif // DG_PATCHES_HPP