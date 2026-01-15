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

/**
 * @brief Detect DG tensors at compile time
 * A DG tensor is a static_tensor with any element type and layout
 */
template <typename T>
struct is_dg_tensor : std::false_type
{
};

template <typename V, typename Layout>
struct is_dg_tensor<amr::containers::static_tensor<V, Layout>> : std::true_type
{
};

template <typename T>
inline constexpr bool is_dg_tensor_v = is_dg_tensor<T>::value;

/**
 * @brief Detect static vectors at compile time
 */
template <typename T>
struct is_static_vector : std::false_type
{
};

template <typename V, std::integral auto N>
struct is_static_vector<amr::containers::static_vector<V, N>> : std::true_type
{
};

template <typename T>
inline constexpr bool is_static_vector_v = is_static_vector<T>::value;

/**
 * @brief Default AMR policy for refinement and coarsening
 * Provides prolongation and restriction operations for adaptive mesh refinement
 */
template <typename global_t, typename Policy>
struct DefaultAMRPolicy
{
    static constexpr std::size_t Dim         = Policy::Dim;
    static constexpr std::size_t NumChildren = (1 << Dim);

    /**
     * @brief Compile-time helper: make offset vector for a given child_offset
     */
    template <std::size_t ChildOffset, std::size_t... Is>
    static constexpr amr::containers::static_vector<double, Dim>
        make_offset_vector_impl(std::index_sequence<Is...>)
    {
        return amr::containers::static_vector<double, Dim>{
            ((ChildOffset >> Is) & 1 ? 0.5 : 0.0)...
        };
    }

    template <std::size_t ChildOffset>
    static constexpr amr::containers::static_vector<double, Dim> make_offset_vector()
    {
        return make_offset_vector_impl<ChildOffset>(std::make_index_sequence<Dim>{});
    }

    /**
     * @brief Compile-time map of all child_offset → offset_vector
     */
    template <std::size_t... Is>
    static constexpr auto make_offset_table_impl(std::index_sequence<Is...>)
    {
        return std::array<amr::containers::static_vector<double, Dim>, NumChildren>{
            make_offset_vector<Is>()...
        };
    }

    static constexpr auto make_offset_table()
    {
        return make_offset_table_impl(std::make_index_sequence<NumChildren>{});
    }

    // The table itself, compile-time
    static constexpr auto offset_table = make_offset_table();

    /**
     * @brief Interpolate values during AMR prolongation
     * Handles three cases: DG tensors, vectors of DG tensors, and everything else
     */
    template <typename Value>
    static constexpr Value interpolate(const Value& parent_val, std::size_t child_offset)
    {
        // Case 1: DG tensor (S1) → real prolongation via basis evaluation
        if constexpr (is_dg_tensor_v<Value>)
        {
            using tensor_t      = Value;
            using multi_index_t = typename tensor_t::multi_index_t;
            using vector_t      = amr::containers::static_vector<double, Dim>;

            Value result{};

            const vector_t& offset = offset_table[child_offset];

            multi_index_t idx{};
            do
            {
                vector_t x{};
                // std::cout << global_t::Quadrature::tensor_point(idx) << " " << offset
                //           << "\n";
                x = 0.5 * global_t::Quadrature::tensor_point(idx) + offset;

                result[idx] = global_t::Basis::evaluate_basis(parent_val, x);
            } while (idx.increment());

            return result;
        }
        // Case 2: Vector of DG tensors (S2 flux) → recursively interpolate each component
        else if constexpr (is_static_vector_v<Value> &&
                           is_dg_tensor_v<typename Value::value_type>)
        {
            Value result{};
            return result;
        }
        // Case 3: Everything else (S3, S4) → zero-initialize (copy parent values)
        else
        {
            return parent_val;
        }
    }

    /**
     * @brief Coarsening (Restriction): compute coarse value from fine children
     * When children cells are coarsened into a parent:
     * parent = child[0]*1 + child[1]*(1/2) + child[2]*(1/3) + child[3]*(1/4) + ...
     * Each child is weighted by 1/(i+1) where i is the child index
     */
    template <typename Container>
    static auto restrict(const Container& fine_children)
    {
        using value_type = typename Container::value_type;
        value_type  coarse_val{};
        std::size_t idx = 0;
        for (const auto& child : fine_children)
        {
            coarse_val = coarse_val + child * (1.0 / static_cast<double>(idx + 1));
            idx++;
        }
        return coarse_val;
    }
};

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
    using tree_t = amr::ndt::tree::ndtree<
        MarkerCell,
        patch_index_t,
        patch_layout_t,
        DefaultAMRPolicy<global_t, Policy>>;

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

                // Zero out S2, S3, S4 - only S1 (dof_patch) is interpolated
                flux_patch[linear_idx] =
                    typename std::remove_reference_t<decltype(flux_patch)>::value_type{};
                size_patch[linear_idx] =
                    typename std::remove_reference_t<decltype(size_patch)>::value_type{};
            }
        }
    };
};

} // namespace amr::dg_tree

#endif // DG_PATCHES_HPP