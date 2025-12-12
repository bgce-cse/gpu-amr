// #ifndef AMR_DG_PATCHES_HPP
// #define AMR_DG_PATCHES_HPP

// #include "containers/container_utils.hpp"
// #include "containers/loop_control.hpp"
// #include "containers/static_layout.hpp"
// #include "containers/static_shape.hpp"
// #include "containers/static_vector.hpp"
// #include "coordinates.hpp"
// #include "morton/morton_id.hpp"
// #include "ndtree/ndtree.hpp"
// #include "ndtree/patch_layout.hpp"

// namespace amr::global
// {

// /**
//  * @brief S1: DOF tensor patch marker
//  * Stores Order^Dim DOF values per cell for polynomial coefficients
//  */
// template <std::size_t Order, unsigned int Dim, std::size_t DOFs>
// struct S1
// {
//     using value_t = amr::containers::static_vector<double, DOFs>;
//     using type    = value_t;

//     static constexpr auto index() noexcept -> std::size_t
//     {
//         return 0;
//     }

//     type data;
// };

// /**
//  * @brief S2: Flux tensor patch marker
//  * Stores Dim flux directions, each with Order^Dim DOFs
//  */
// template <std::size_t Order, unsigned int Dim, std::size_t DOFs>
// struct S2
// {
//     using value_t = amr::containers::static_vector<double, DOFs>;
//     using type    = amr::containers::static_vector<value_t, Dim>;

//     static constexpr auto index() noexcept -> std::size_t
//     {
//         return 1;
//     }

//     type data;
// };

// /**
//  * @brief S3: Cell center coordinates patch marker
//  */
// template <unsigned int Dim>
// struct S3
// {
//     using type = amr::containers::static_vector<double, Dim>;

//     static constexpr auto index() noexcept -> std::size_t
//     {
//         return 2;
//     }

//     type data;
// };

// /**
//  * @brief S4: Cell size/volume patch marker
//  */
// template <unsigned int Dim>
// struct S4
// {
//     using type = amr::containers::static_vector<double, Dim>;

//     static constexpr auto index() noexcept -> std::size_t
//     {
//         return 3;
//     }

//     type data;
// };

// /**
//  * @brief Marker cell type for tree AMR with DG data
//  */
// template <std::size_t Order, unsigned int Dim, std::size_t DOFs>
// struct MarkerCell
// {
//     using deconstructed_types_map_t =
//         std::tuple<S1<Order, Dim, DOFs>, S2<Order, Dim, DOFs>, S3<Dim>, S4<Dim>>;

//     MarkerCell() = default;

//     auto data_tuple() -> auto&
//     {
//         return m_data;
//     }

//     auto data_tuple() const -> auto const&
//     {
//         return m_data;
//     }

//     deconstructed_types_map_t m_data;
// };

// template <
//     std::size_t PatchSize,
//     std::size_t HaloWidth,
//     // std::size_t Dim,
//     std::size_t Order,
//     std::size_t NumDOF,
//     std::size_t MaxDepth,
//     typename EquationType>
// struct TreeInitializer
// {
//     using shape_t        = amr::containers::static_shape<PatchSize, PatchSize>;
//     using layout_t       = amr::containers::static_layout<shape_t>;
//     using marker_cell_t  = MarkerCell<Order, Dim, NumDOF>;
//     using patch_index_t  = amr::ndt::morton::morton_id<MaxDepth, Dim>;
//     using patch_layout_t = amr::ndt::patches::patch_layout<layout_t, HaloWidth>;
//     using tree_t = amr::ndt::tree::ndtree<marker_cell_t, patch_index_t,
//     patch_layout_t>;

//     // Tree member initialized with default capacity
//     tree_t tree{ 10000 };

//     // Constructor: initializes tree and fills every patch with equation data
//     TreeInitializer()
//     {
//         // Iterate through all patches in the tree
//         for (std::size_t idx = 0; idx < tree.size(); ++idx)
//         {
//             auto& dof_patch  = tree.template get_patch<S1<Order, Dim, NumDOF>>(idx);
//             auto& flux_patch = tree.template get_patch<S2<Order, Dim, NumDOF>>(idx);
//             auto& center_coord_patch = tree.template get_patch<S3<Dim>>(idx);
//             auto& size_patch         = tree.template get_patch<S4<Dim>>(idx);

//             auto patch_id                    = patch_index_t(idx);
//             auto [patch_coords, patch_level] = patch_index_t::decode(patch_id.id());
//             double patch_level_size = 1.0 / static_cast<double>(1u << patch_level);
//             double cell_size        = patch_level_size /
//             static_cast<double>(PatchSize);

//             // Initialize each cell in the patch
//             for (std::size_t linear_idx = 0; linear_idx != patch_layout_t::flat_size();
//                  ++linear_idx)
//             {
//                 // Skip halo cells
//                 if (amr::ndt::utils::patches::is_halo_cell<patch_layout_t>(linear_idx))
//                     continue;

//                 // Compute cell center using existing coordinate helper
//                 auto coords_with_halo =
//                     linear_to_local_coords<Dim, PatchSize, HaloWidth>(linear_idx);
//                 auto local_indices = remove_halo<Dim, HaloWidth>(coords_with_halo);
//                 auto cell_center   = compute_cell_center<PatchSize, HaloWidth, Dim>(
//                     patch_id, local_indices
//                 );
//                 center_coord_patch[linear_idx] = cell_center;

//                 // Initialize S1 (DOF tensor) using equation's initial condition
//                 dof_patch[linear_idx] = EquationType::get_initial_values(cell_center);

//                 // Initialize S2 (flux tensor) using equation's flux evaluation
//                 flux_patch[linear_idx] =
//                     EquationType::evaluate_flux(dof_patch[linear_idx]);

//                 for (std::size_t d = 0; d < Dim; ++d)
//                 {
//                     size_patch[linear_idx][d] = cell_size;
//                 }
//             }
//         }
//     }
// };

// } // namespace amr::global

// #endif // AMR_DG_PATCHES_HPP
