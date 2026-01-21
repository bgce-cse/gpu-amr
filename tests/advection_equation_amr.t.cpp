#include <gtest/gtest.h>
#include "containers/static_layout.hpp"
#include "containers/static_shape.hpp"
#include "containers/static_vector.hpp"
#include "morton/morton_id.hpp"
#include "ndtree/ndhierarchy.hpp"
#include "ndtree/ndtree.hpp"
#include "ndtree/patch_layout.hpp"
#include "solver/amr_solver.hpp"
#include "solver/cell_types.hpp"
#include "solver/physics_system.hpp"
#include "solver/AdvectionPhysics.hpp"


template <int Dim, std::size_t PatchDim>
struct AdvectionTestConfig {
    static constexpr int DIM = Dim;
    
    // Geometry/Layout types
    using shape_t = amr::containers::static_shape<PatchDim, PatchDim>;
    using layout_t = amr::containers::static_layout<shape_t>;
    using patch_index_t = amr::ndt::morton::morton_id<7u, 2u>;
    using patch_layout_t = amr::ndt::patches::patch_layout<layout_t, 2>;
    
    // Physics/Tree types
    using equation_t = AdvectionPhysics<Dim>;
    using cell_t = amr::cell::AdvectionCell;
    using tree_t = amr::ndt::tree::ndtree<cell_t, patch_index_t, patch_layout_t>;
    
    // Geometry logic
    static constexpr std::array<double, Dim> domain_lengths = {1.0, 1.0};
    using geometry_t = amr::ndt::solver::physics_system<patch_index_t, patch_layout_t, domain_lengths>;
};

using AdvectionTestTypes = ::testing::Types<
    AdvectionTestConfig<2, 10>,
    AdvectionTestConfig<2, 50>
>;

template <typename T>
class AdvectionSolverTest : public ::testing::Test {};

TYPED_TEST_SUITE(AdvectionSolverTest, AdvectionTestTypes);

TYPED_TEST(AdvectionSolverTest, PulseCenterMovement)
{
    using Config = TypeParam;
    
    // Setup Solver
    amr_solver<typename Config::tree_t, typename Config::geometry_t, typename Config::equation_t, Config::DIM> 
        solver(2000, 1.0, 0.4);

    // Initial pulse at (0.2, 0.2)
    const std::array<double, 2> start_pos = {0.2, 0.2};
    auto ic = [&](auto const& coords) -> amr::containers::static_vector<double, 1> {
        double r2 = 0.0;
        for(int d=0; d<Config::DIM; ++d) r2 += std::pow(coords[d] - start_pos[d], 2);
        return { std::exp(-r2 / 0.005) };
    };

    solver.initialize(ic);
    
    // Run for a fixed amount of time
    double t = 0.0;
    double t_target = 0.5;
    while (t < t_target) {
        double dt = solver.compute_time_step();
        if (t + dt > t_target) dt = t_target - t;
        solver.time_step(dt);
        solver.get_tree().halo_exchange_update();
        t += dt;
    }

    // Expected final position: x_new = x_old + v * t
    std::array<double, Config::DIM> expected_pos;
    for(int d=0; d < Config::DIM; ++d) {
        expected_pos[d] = start_pos[d] + Config::equation_t::Velocity[d] * t_target;
    }

    // Find the cell with the maximum value (the new center)
    double max_val = -1.0;
    std::array<double, Config::DIM> actual_pos{};

    auto& tree = solver.get_tree();
    for (std::size_t p_idx = 0; p_idx < tree.size(); ++p_idx) {
        auto patch_id = tree.get_node_index_at(p_idx);
        auto const& patch = tree.template get_patch<amr::cell::Scalar>(p_idx);
        
        for (std::size_t l_idx = 0; l_idx < Config::patch_layout_t::flat_size(); ++l_idx) {
            if (amr::ndt::utils::patches::is_halo_cell<typename Config::patch_layout_t>(l_idx)) continue;
            
            if (patch[l_idx] > max_val) {
                max_val = patch[l_idx];
                auto coord = Config::geometry_t::cell_coord(patch_id, l_idx);
                auto size = Config::geometry_t::cell_sizes(patch_id);
                for(int d=0; d<Config::DIM; ++d) actual_pos[d] = coord[d] + 0.5 * size[d];
            }
        }
    }

    // Verify coordinates with a tolerance (half a cell width)
    auto cell_size = Config::geometry_t::cell_sizes(tree.get_node_index_at(0));
    for(int d=0; d < Config::DIM; ++d) {
        std::cout << "Expected pos (" << d << "): " << expected_pos[d] << ", Actual pos (" << d << "): " << actual_pos[d] << std::endl;
        EXPECT_NEAR(actual_pos[d], expected_pos[d], cell_size[d]) 
            << "Pulse center mismatch in dimension " << d;
    }
}

TYPED_TEST(AdvectionSolverTest, PulseCenterMovementAMR)
{
    using Config = TypeParam;
    
    // Setup Solver
    amr_solver<typename Config::tree_t, typename Config::geometry_t, typename Config::equation_t, Config::DIM> 
        solver(2000, 1.0, 0.4);

    // Initial pulse at (0.2, 0.2)
    const std::array<double, 2> start_pos = {0.2, 0.2};
    auto ic = [&](auto const& coords) -> amr::containers::static_vector<double, 1> {
        double r2 = 0.0;
        for(int d=0; d<Config::DIM; ++d) r2 += std::pow(coords[d] - start_pos[d], 2);
        return { std::exp(-r2 / 0.005) };
    };

    // AMR Criterion
    auto amrCriterion = [&](const Config::patch_index_t& idx) {
        auto patch = solver.get_tree().template get_patch<amr::cell::Scalar>(idx);
        auto& data = patch.data();
        double max_val = 0;
        for (auto v : data) {
            max_val = std::max(max_val, v);
        }

        if (max_val > 0.1 && idx.level() < 5) return Config::tree_t::refine_status_t::Refine;
        if (max_val < 0.05 && idx.level() > 1) return Config::tree_t::refine_status_t::Coarsen;
        return Config::tree_t::refine_status_t::Stable;
    };

    solver.initialize(ic);
    solver.get_tree().halo_exchange_update();
    
    // Run for a fixed amount of time
    double t = 0.0;
    double t_target = 0.5;
    while (t < t_target) {
        double dt = solver.compute_time_step();
        if (t + dt > t_target) dt = t_target - t;
        solver.time_step(dt);
        solver.get_tree().halo_exchange_update();
        solver.get_tree().reconstruct_tree(amrCriterion);
        solver.get_tree().halo_exchange_update();
        t += dt;
    }

    // Expected final position: x_new = x_old + v * t
    std::array<double, Config::DIM> expected_pos;
    for(int d=0; d < Config::DIM; ++d) {
        expected_pos[d] = start_pos[d] + Config::equation_t::Velocity[d] * t_target;
    }

    // Find the cell with the maximum value (the new center)
    double max_val = -1.0;
    std::array<double, Config::DIM> actual_pos{};

    auto& tree = solver.get_tree();
    for (std::size_t p_idx = 0; p_idx < tree.size(); ++p_idx) {
        auto patch_id = tree.get_node_index_at(p_idx);
        auto const& patch = tree.template get_patch<amr::cell::Scalar>(p_idx);
        
        for (std::size_t l_idx = 0; l_idx < Config::patch_layout_t::flat_size(); ++l_idx) {
            if (amr::ndt::utils::patches::is_halo_cell<typename Config::patch_layout_t>(l_idx)) continue;
            
            if (patch[l_idx] > max_val) {
                max_val = patch[l_idx];
                auto coord = Config::geometry_t::cell_coord(patch_id, l_idx);
                auto size = Config::geometry_t::cell_sizes(patch_id);
                for(int d=0; d<Config::DIM; ++d) actual_pos[d] = coord[d] + 0.5 * size[d];
            }
        }
    }

    // Verify coordinates with a tolerance (half a cell width)
    auto cell_size = Config::geometry_t::cell_sizes(tree.get_node_index_at(0));
    for(int d=0; d < Config::DIM; ++d) {
        std::cout << "Expected pos (" << d << "): " << expected_pos[d] << ", Actual pos with AMR (" << d << "): " << actual_pos[d] << std::endl;
        EXPECT_NEAR(actual_pos[d], expected_pos[d], cell_size[d]) 
            << "Pulse center mismatch in dimension " << d;
    }
}

