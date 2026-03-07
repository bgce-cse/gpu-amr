#ifndef AMR_GLOBAL_AMR_INDICATORS_HPP
#define AMR_GLOBAL_AMR_INDICATORS_HPP

#include "coordinates.hpp"
#include "generated_config.hpp"
#include "ndtree/patch_utils.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace amr::global
{

// =====================================================================
//  Thresholds for gradient-based AMR indicators
// =====================================================================
struct AMRThresholds
{
    double refine_threshold  = 0.5;  ///< Cell indicator > this  → vote Refine
    double coarsen_threshold = 0.05; ///< Cell indicator < this  → vote Coarsen
};

// =====================================================================
//  Variation-based AMR indicator
//
//  For every **leaf patch** in the tree the indicator
//
//    1.  inspects each interior cell's DG tensor internally,
//    2.  measures the spread (max − min) of the first Dim DOF
//        components (momentum / velocity in each direction)
//        across the quadrature nodes within the cell,
//    3.  normalises by the cell-mean density to obtain a velocity
//        variation,
//    4.  if **any** interior cell exceeds the refine threshold the
//        whole patch is refined (early exit — no need to check the
//        remaining cells),
//    5.  if **all** interior cells fall below the coarsen threshold
//        the patch is coarsened,
//    6.  otherwise the patch stays Stable.
//
//  No neighbor access is required — the indicator is purely local to
//  each DG cell.
//
//  The result is a  std::vector<refine_status_t>  whose indices correspond
//  to the "natural" sequential patch layout  [0 .. tree.size()-1].
//  The solver can feed this directly into  reconstruct_tree()  via
//  a simple look-up lambda.
//
//  Template parameters
//  -------------------
//  @tparam global_t   Fully assembled GlobalConfig (coordinate helpers etc.)
//  @tparam Policy     Compile-time config policy
//                     (PatchSize, HaloWidth, Dim, DOFs, Order, …)
// =====================================================================
template <typename global_t, typename Policy>
struct GradientAMRIndicator
{
    static constexpr std::size_t Dim        = Policy::Dim;
    static constexpr std::size_t Order      = Policy::Order;
    static constexpr std::size_t DOFs       = Policy::DOFs;
    static constexpr std::size_t PatchSize  = Policy::PatchSize;
    static constexpr std::size_t HaloWidth  = Policy::HaloWidth;
    static constexpr std::size_t PaddedSize = PatchSize + 2 * HaloWidth;

    // -----------------------------------------------------------------
    //  cell_average
    //
    //  Compute the arithmetic mean of the DOF nodal values within a
    //  single DG cell.  Returns a vector of length DOFs (one average per
    //  conservative variable: ρ, ρu, ρv, E, …).
    //
    //  @param dof_tensor  The DG nodal tensor stored at one cell
    //                     (Order^Dim entries, each a static_vector<double,DOFs>)
    // -----------------------------------------------------------------
    template <typename DofTensor>
    static auto cell_average(const DofTensor& dof_tensor)
        -> amr::containers::static_vector<double, DOFs>
    {
        amr::containers::static_vector<double, DOFs> avg{};
        for (auto& a : avg)
            a = 0.0;

        std::size_t count = 0;

        using multi_index_t = typename DofTensor::multi_index_t;
        multi_index_t idx{};
        do
        {
            const auto& nodal_val = dof_tensor[idx];
            for (std::size_t v = 0; v < DOFs; ++v)
                avg[v] += nodal_val[v];
            ++count;
        } while (idx.increment());

        if (count > 0)
        {
            const double inv_count = 1.0 / static_cast<double>(count);
            for (std::size_t v = 0; v < DOFs; ++v)
                avg[v] *= inv_count;
        }
        return avg;
    }

    // -----------------------------------------------------------------
    //  compute_cell_indicator
    //
    //  Internal variation indicator (no neighbor access).
    //
    //  For each spatial direction d (0..Dim-1) the indicator measures
    //  the spread of DOF component d (= momentum rho*u_d) across all
    //  quadrature nodes inside the DG tensor of a single cell:
    //
    //      variation_d = (max - min) of nodal_val[d]  over all nodes
    //
    //  This is normalised by the cell-mean density so the result is
    //  a velocity-like variation.  The cell indicator is the max over
    //  all Dim directions.
    //
    //  @param dof_tensor  The DG nodal tensor at one cell
    //  @return            Non-negative indicator value
    // -----------------------------------------------------------------
    template <typename DofTensor>
    static auto compute_cell_indicator_internal(const DofTensor& dof_tensor) -> double
    {
        using multi_index_t = typename DofTensor::multi_index_t;

        // Advection: track only the first DOF (variable 0).
        // Euler:     momentum variation normalized by cell-mean density.
        if constexpr (Policy::equation == amr::config::EquationType::Advection)
        {
            double vmin = std::numeric_limits<double>::max();
            double vmax = std::numeric_limits<double>::lowest();

            multi_index_t idx{};
            do
            {
                const double val = static_cast<double>(dof_tensor[idx][0]);
                vmin             = std::min(vmin, val);
                vmax             = std::max(vmax, val);
            } while (idx.increment());

            return vmax - vmin;
        }
        else
        {
            // Euler layout: [mom_0, ..., mom_{Dim-1}, rho, E]
            constexpr std::size_t DENSITY_IDX = Dim;
            constexpr double      eps         = 1.0e-10;

            std::array<double, Dim> vmin{};
            std::array<double, Dim> vmax{};
            for (std::size_t d = 0; d < Dim; ++d)
            {
                vmin[d] = std::numeric_limits<double>::max();
                vmax[d] = std::numeric_limits<double>::lowest();
            }
            double      rho_sum = 0.0;
            std::size_t count   = 0;

            multi_index_t idx{};
            do
            {
                const auto& nodal_val = dof_tensor[idx];
                for (std::size_t d = 0; d < Dim; ++d)
                {
                    vmin[d] = std::min(vmin[d], static_cast<double>(nodal_val[d]));
                    vmax[d] = std::max(vmax[d], static_cast<double>(nodal_val[d]));
                }
                rho_sum += static_cast<double>(nodal_val[DENSITY_IDX]);
                ++count;
            } while (idx.increment());

            const double mean_rho =
                (count > 0) ? rho_sum / static_cast<double>(count) : 1.0;
            const double inv_rho = 1.0 / (std::abs(mean_rho) + eps);

            double indicator = 0.0;
            for (std::size_t d = 0; d < Dim; ++d)
                indicator = std::max(indicator, (vmax[d] - vmin[d]) * inv_rho);

            return indicator;
        }
    }

    // -----------------------------------------------------------------
    //  compute_patch_decisions
    //
    //  Main entry point.  Iterates over every leaf patch in the tree.
    //  If ANY interior cell exceeds the refine threshold the patch is
    //  refined (early exit).  If ALL interior cells are below the
    //  coarsen threshold the patch is coarsened.  Otherwise Stable.
    //
    //  @tparam TreeT   ndtree type
    //  @tparam S1Tag   The DOF map-type tag (typically TreeBuilder::S1)
    //
    //  @param tree         The ndtree holding all leaf patches
    //  @param thresholds   User-specified refine / coarsen thresholds
    //
    //  @return  std::vector<refine_status_t>  of length  tree.size().
    //           Entry i is the AMR decision for the patch at sequential
    //           index i in the natural layout.
    // -----------------------------------------------------------------
    template <typename TreeT, typename S1Tag>
    static auto
        compute_patch_decisions(const TreeT& tree, const AMRThresholds& thresholds)
            -> std::vector<typename TreeT::refine_status_t>
    {
        using refine_status_t = typename TreeT::refine_status_t;
        using patch_layout_t  = typename TreeT::patch_layout_t;

        const std::size_t            num_patches = tree.size();
        std::vector<refine_status_t> decisions(num_patches, refine_status_t::Stable);

        for (std::size_t p = 0; p < num_patches; ++p)
        {
            const auto& dof_patch   = tree.template get_patch<S1Tag>(p);
            bool        all_coarsen = true;

            for (std::size_t l_idx = 0; l_idx < patch_layout_t::flat_size(); ++l_idx)
            {
                if (amr::ndt::utils::patches::is_halo_cell<patch_layout_t>(l_idx))
                    continue;

                const double indicator =
                    compute_cell_indicator_internal(dof_patch[l_idx]);

                if (indicator > thresholds.refine_threshold)
                {
                    decisions[p] = refine_status_t::Refine;
                    all_coarsen  = false;
                    break; // one cell is enough
                }
                if (indicator >= thresholds.coarsen_threshold) all_coarsen = false;
            }

            if (decisions[p] != refine_status_t::Refine && all_coarsen)
                decisions[p] = refine_status_t::Coarsen;
        }

        return decisions;
    }
};

} // namespace amr::global

#endif // AMR_GLOBAL_AMR_INDICATORS_HPP
