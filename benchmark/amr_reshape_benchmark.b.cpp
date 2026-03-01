#include "containers/static_layout.hpp"
#include "containers/static_shape.hpp"
#include "containers/static_vector.hpp"
#include "morton/morton_id.hpp"
#include "ndtree/ndhierarchy.hpp"
#include "ndtree/ndtree.hpp"
#include "ndtree/patch_layout.hpp"
#include "solver/cell_types.hpp"
#include "utility/logging.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <numeric>
#include <vector>

constexpr std::size_t N    = 10;
constexpr std::size_t M    = 10;
constexpr std::size_t Halo = 2;

using shape_t        = amr::containers::static_shape<N, M>;
using layout_t       = amr::containers::static_layout<shape_t>;
using patch_index_t  = amr::ndt::morton::morton_id<8u, 2u>;
using patch_layout_t = amr::ndt::patches::patch_layout<layout_t, Halo>;
using tree_t         = amr::ndt::tree::ndtree<amr::cell::EulerCell2D, patch_index_t, patch_layout_t>;
using status_t       = tree_t::RefinementStatus;

using bench_clock_t = std::chrono::steady_clock;
using duration_t    = std::chrono::duration<double, std::micro>;

static auto now() { return bench_clock_t::now(); }
static auto elapsed_us(bench_clock_t::time_point start) -> double
{
    return duration_t(bench_clock_t::now() - start).count();
}

static auto make_uniform_tree(int depth) -> tree_t
{
    const std::size_t capacity = static_cast<std::size_t>(1) << (depth * 2 + 4);
    tree_t tree{ capacity };

    for (int d = 0; d < depth; ++d)
    {
        tree.update_refine_flags([](auto const&) { return status_t::Refine; });
        tree.apply_refine_coarsen();
        tree.balancing();
        tree.fragment();
    }
    return tree;
}

static void bench_fragment_recombine(int depth, int repetitions)
{
    std::vector<double> times_frag(repetitions);
    std::vector<double> times_rec(repetitions);
    std::size_t n_patches_before_frag = 0;
    std::size_t n_patches_before_rec  = 0;

    for (int r = 0; r < repetitions; ++r)
    {
        {
            auto tree             = make_uniform_tree(depth);
            n_patches_before_frag = tree.size();

            tree.update_refine_flags([](auto const&) { return status_t::Refine; });
            tree.apply_refine_coarsen();
            tree.balancing();

            auto t0       = now();
            tree.fragment();
            times_frag[r] = elapsed_us(t0);
        }

        {
            auto tree            = make_uniform_tree(depth + 1);
            n_patches_before_rec = tree.size();

            tree.update_refine_flags([depth](auto const& idx)
            {
                return patch_index_t::level(idx) == depth + 1
                    ? status_t::Coarsen
                    : status_t::Stable;
            });
            tree.apply_refine_coarsen();
            tree.balancing();

            auto t1      = now();
            tree.recombine();
            times_rec[r] = elapsed_us(t1);
        }
    }

    auto stats = [&](std::vector<double> const& v)
    {
        double mean = std::accumulate(v.begin(), v.end(), 0.0) / repetitions;
        double mn   = *std::min_element(v.begin(), v.end());
        double mx   = *std::max_element(v.begin(), v.end());
        return std::tuple{ mean, mn, mx };
    };

    auto [fmean, fmn, fmx] = stats(times_frag);
    auto [rmean, rmn, rmx] = stats(times_rec);

    std::printf(
        "[fragment      ] depth=%d  patches_in=%6zu  mean=%8.2f us  min=%8.2f us  max=%8.2f us\n",
        depth, n_patches_before_frag, fmean, fmn, fmx);
    std::printf(
        "[recombine     ] depth=%d  patches_in=%6zu  mean=%8.2f us  min=%8.2f us  max=%8.2f us\n",
        depth, n_patches_before_rec, rmean, rmn, rmx);
}

static void bench_full_amr_cycle(int depth, int repetitions)
{
    auto tree = make_uniform_tree(depth);

    std::vector<double> times(repetitions);
    for (int r = 0; r < repetitions; ++r)
    {
        auto t0 = now();
        tree.reconstruct_tree([depth](auto const& idx)
        {
            const auto lvl = patch_index_t::level(idx);
            if (lvl < depth)  return status_t::Refine;
            if (lvl == depth) return status_t::Stable;
            return status_t::Coarsen;
        });
        times[r] = elapsed_us(t0);
    }

    const double mean = std::accumulate(times.begin(), times.end(), 0.0) / repetitions;
    const double mn   = *std::min_element(times.begin(), times.end());
    const double mx   = *std::max_element(times.begin(), times.end());

    std::printf(
        "[full_amr_cycle] depth=%d  patches=%6zu  mean=%8.2f us  min=%8.2f us  max=%8.2f us\n",
        depth, tree.size(), mean, mn, mx);
}

int main()
{
    constexpr int repetitions = 50;

    std::printf("=== AMR reshape benchmark ===\n");
    std::printf("patch_flat_size = %u  |  repetitions = %d\n\n",
        static_cast<unsigned>(patch_layout_t::flat_size()), repetitions);

    for (int depth : { 2, 3, 4, 5 })
    {
        bench_fragment_recombine(depth, repetitions);
        bench_full_amr_cycle(depth, repetitions);
        std::printf("\n");
    }

    return 0;
}