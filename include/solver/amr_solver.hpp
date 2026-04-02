#ifndef AMR_SOLVER_HPP
#define AMR_SOLVER_HPP

#include "EulerPhysics.hpp"
#include "cell_types.hpp"
#include "config/definitions.hpp"
#include "containers/container_manipulations.hpp"
#include "ndtree/ndtree.hpp"
#include "physics_system.hpp"
#ifdef AMR_ENABLE_CUDA_AMR
#include "cuda/device_buffer.hpp"
#include "cuda/fvm_time_step.hpp"
#include "cuda/profiler.hpp"
#endif
#include <algorithm>
#include <cstdint>
#include <execution>
#include <limits>
#include <vector>

/**
 * @brief AMR Solver
 * @tparam TreeT The ndtree structure
 * @tparam GeometryT The physics_system (coordinates/sizes)
 * @tparam EquationT The physical model (Euler, Advection, etc.)
 * @tparam DIM Dimensionality
 */
template <typename TreeT, typename GeometryT, typename EquationT, int DIM>
class amr_solver
{
public:
    using tree_t               = TreeT;
    using patch_layout_t       = typename tree_t::patch_layout_t;
    using patch_index_t        = typename tree_t::patch_index_t;
    using linear_index_t       = typename tree_t::linear_index_t;
    using data_layout_t        = typename patch_layout_t::data_layout_t;
    using padded_layout_t      = typename patch_layout_t::padded_layout_t;
    static constexpr auto NVAR = EquationT::NVAR;
    // TODO: Make a template
    using arithmetic_t = double;

private:
    TreeT              m_tree;
    arithmetic_t const m_gamma; // Specific heat ratio
    arithmetic_t const m_cfl;   // CFL number
    arithmetic_t       m_host_pending_batch_dt{};
    std::size_t        m_host_pending_batch_step_count{};
#ifdef AMR_ENABLE_CUDA_AMR
    mutable arithmetic_t*   m_device_dt_buffer                = nullptr;
    mutable arithmetic_t* m_device_dt_accumulator_buffer = nullptr;
    mutable arithmetic_t*   m_device_remaining_time_buffer    = nullptr;
    mutable arithmetic_t*   m_pinned_dt_buffer                = nullptr;
    mutable std::uint32_t*  m_device_executed_step_count_buffer = nullptr;
    mutable std::uint32_t*  m_pinned_executed_step_count_buffer = nullptr;
    mutable void*           m_dt_ready_fence                  = nullptr;
    mutable void*           m_dt_copy_fence                   = nullptr;
    mutable void*           m_dt_copy_stream                  = nullptr;
    mutable std::size_t     m_dt_copy_pending                 = 0;
#endif

public:
    amr_solver(size_t capacity, arithmetic_t gamma_ = 1.4, arithmetic_t cfl_ = 0.1)
        : m_tree(capacity)
        , m_gamma(gamma_)
        , m_cfl(cfl_)
    {
        // Dummy dimensions were removed from new EulerPhysics.hpp
        static_assert(DIM == 2 || DIM == 3, "Error: Wrong dimension");
    }

    ~amr_solver()
    {
#ifdef AMR_ENABLE_CUDA_AMR
        if (m_dt_copy_pending != 0)
        {
            amr::cuda::async_copy_fence_wait(m_dt_copy_fence);
        }
        amr::cuda::async_copy_stream_destroy(m_dt_copy_stream);
        amr::cuda::async_copy_fence_destroy(m_dt_ready_fence);
        amr::cuda::async_copy_fence_destroy(m_dt_copy_fence);
        amr::cuda::host_pinned_free(static_cast<void*>(m_pinned_executed_step_count_buffer));
        amr::cuda::host_pinned_free(static_cast<void*>(m_pinned_dt_buffer));
        amr::cuda::device_free(static_cast<void*>(m_device_executed_step_count_buffer));
        amr::cuda::device_free(static_cast<void*>(m_device_remaining_time_buffer));
        amr::cuda::device_free(static_cast<void*>(m_device_dt_accumulator_buffer));
        amr::cuda::device_free(static_cast<void*>(m_device_dt_buffer));
#endif
    }

    TreeT& get_tree()
    {
        return m_tree;
    }

    arithmetic_t get_gamma() const noexcept
    {
        return m_gamma;
    }

    arithmetic_t get_cfl() const noexcept
    {
        return m_cfl;
    }

    void initialize(auto&& init_func)
    {
        for (std::size_t p_idx = 0; p_idx < m_tree.size(); ++p_idx)
        {
            const auto patch_id = m_tree.get_node_index_at(p_idx);
            const auto c_size   = GeometryT::cell_sizes(patch_id);
            // Fetch the tuple of SoA patch references
            auto fetch_patches = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                return std::forward_as_tuple(
                    m_tree.template get_patch<
                        typename std::tuple_element<Is, typename EquationT::FieldTags>::type
                    >(p_idx)...
                );
            };
            auto patches = fetch_patches(std::make_index_sequence<NVAR>{});
            amr::containers::manipulators::shaped_for<
                typename patch_layout_t::interior_iteration_control_t>(
                [this, patch_id, &c_size, &patches](auto&& fn, auto const& idxs)
                {
                    const auto linear_idx = padded_layout_t::linear_index(idxs);
        
                    // Dimension-agnostic coordinate fetch
                    const auto cell_origin = GeometryT::cell_coord(patch_id, linear_idx);
                    std::array<arithmetic_t, DIM> coords;
                    for (int d = 0; d < DIM; ++d)
                        coords[d] = cell_origin[d] + 0.5 * c_size[d];
                    // IC -> Primitive -> Conservative
                    const auto prim = std::invoke(std::forward<decltype(fn)>(fn), coords);
                    amr::containers::static_vector<arithmetic_t, NVAR> cons;
                    EquationT::primitiveToConservative(prim, cons, m_gamma);
                    // Write directly to the SoA arrays
                    auto write_state = [&]<std::size_t... Is>(std::index_sequence<Is...>) -> void
                    {
                        ((std::get<Is>(patches)[linear_idx] = cons[Is]), ...);
                    };
                    write_state(std::make_index_sequence<NVAR>{});
                },
                std::forward<decltype(init_func)>(init_func)
            );
        }
    }

    template <auto B> struct BufferTag { static constexpr auto value = B; };

    auto advance() -> arithmetic_t
    {
        advance_batch_async(1);
        return finish_advance_batch();
    }

    auto advance_batch_async(
        std::size_t step_count,
        arithmetic_t remaining_time = std::numeric_limits<arithmetic_t>::max()
    ) -> void
    {
#ifdef AMR_ENABLE_CUDA_AMR
        auto nvtx_range = amr::cuda::scoped_profile_range{ "advance_batch_async" };
        ensure_cuda_dt_resources();
        wait_for_pending_dt_copy();

        amr::cuda::launch_set_double_buffer(m_device_dt_accumulator_buffer, 0.0);
        amr::cuda::launch_set_double_buffer(m_device_remaining_time_buffer, remaining_time);
        amr::cuda::launch_set_uint32_buffer(m_device_executed_step_count_buffer, 0u);

        auto config = make_cuda_time_step_config();
        config.cfl  = m_cfl;

        for (std::size_t step_idx = 0; step_idx < step_count; ++step_idx)
        {
            auto in_ptrs    = get_device_ro_ptrs(BufferTag<tree_t::current_buffer>{});
            auto out_ptrs   = get_device_rw_ptrs(BufferTag<tree_t::next_buffer>{});
            auto in_ptrs_rw = in_ptrs_as_mutable(in_ptrs);

            amr::cuda::launch_compute_dt_kernel_device<EquationT, DIM>(
                in_ptrs,
                m_tree.get_device_patch_level_buffer(),
                config,
                m_device_dt_buffer
            );

            amr::cuda::launch_finalize_step_dt(
                m_device_dt_buffer,
                m_device_dt_accumulator_buffer,
                m_device_remaining_time_buffer,
                m_device_executed_step_count_buffer,
                m_cfl
            );

            amr::cuda::launch_time_step_kernel_with_device_dt<EquationT, DIM>(
                in_ptrs_rw,
                out_ptrs,
                m_tree.get_device_patch_level_buffer(),
                config,
                m_device_dt_buffer
            );

            m_tree.swap_buffers();
            m_tree.halo_exchange_update();
        }

        amr::cuda::async_copy_fence_record(m_dt_ready_fence);
        amr::cuda::async_copy_stream_wait_for_fence(m_dt_copy_stream, m_dt_ready_fence);
        amr::cuda::copy_device_to_host_async_on_stream(
            static_cast<void*>(m_pinned_dt_buffer),
            static_cast<void const*>(m_device_dt_accumulator_buffer),
            sizeof(arithmetic_t),
            m_dt_copy_stream
        );
        amr::cuda::copy_device_to_host_async_on_stream(
            static_cast<void*>(m_pinned_executed_step_count_buffer),
            static_cast<void const*>(m_device_executed_step_count_buffer),
            sizeof(std::uint32_t),
            m_dt_copy_stream
        );
        amr::cuda::async_copy_fence_record_on_stream(m_dt_copy_fence, m_dt_copy_stream);
        m_dt_copy_pending = 1;
#else
        m_host_pending_batch_dt = 0.0;
        m_host_pending_batch_step_count = 0;
        auto remaining = remaining_time;
        for (std::size_t step_idx = 0; step_idx < step_count; ++step_idx)
        {
            if (remaining <= 0.0)
            {
                break;
            }
            const auto dt = compute_time_step_cpu();
            const auto step_dt = std::min(dt, remaining);
            if (step_dt <= 0.0)
            {
                break;
            }
            time_step_cpu(step_dt);
            m_host_pending_batch_dt += step_dt;
            remaining -= step_dt;
            ++m_host_pending_batch_step_count;
        }
#endif
    }

    auto finish_advance_batch(std::size_t* executed_step_count = nullptr) -> arithmetic_t
    {
#ifdef AMR_ENABLE_CUDA_AMR
        auto nvtx_range = amr::cuda::scoped_profile_range{ "finish_advance_batch" };
        wait_for_pending_dt_copy();
        if (executed_step_count != nullptr)
        {
            *executed_step_count = static_cast<std::size_t>(*m_pinned_executed_step_count_buffer);
        }
        return *m_pinned_dt_buffer;
#else
        if (executed_step_count != nullptr)
        {
            *executed_step_count = m_host_pending_batch_step_count;
        }
        return m_host_pending_batch_dt;
#endif
    }

private:
    auto time_step_cpu(const arithmetic_t dt) -> void
    {
        static constexpr auto stride_y =
            patch_layout_t::padded_layout_t::shape_t::sizes()[DIM - 1];
        static constexpr auto stride_z =
            (DIM == 3) ? patch_layout_t::padded_layout_t::shape_t::sizes()[1] * stride_y
                       : 0;

        auto const r = std::views::iota(decltype(m_tree.size()){}, m_tree.size());
        std::for_each(
            AMR_EXECUTION_POLICY,
            std::cbegin(r),
            std::cend(r),
            [this, dt](auto const p_idx) mutable
            {
                const auto patch_id = m_tree.get_node_index_at(p_idx);
                const auto c_size   = GeometryT::cell_sizes(patch_id);

                auto fetch_in_patches = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                    return std::forward_as_tuple(
                        m_tree.template get_patch<
                            typename std::tuple_element<Is, typename EquationT::FieldTags>::type
                        >(p_idx)...
                    );
                };
                auto in_patches = fetch_in_patches(std::make_index_sequence<NVAR>{});

                auto fetch_out_patches = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                    return std::forward_as_tuple(
                        m_tree.template get_out_patch<
                            typename std::tuple_element<Is, typename EquationT::FieldTags>::type,
                            tree_t::next_buffer
                        >(p_idx)...
                    );
                };
                auto out_patches = fetch_out_patches(std::make_index_sequence<NVAR>{});

                amr::containers::manipulators::shaped_for<
                    typename patch_layout_t::interior_iteration_control_t>(
                    [this, dt, &c_size, &in_patches, &out_patches](
                        auto const& idxs
                    )
                    {
                        const auto linear_idx = padded_layout_t::linear_index(idxs);
                        amr::containers::static_vector<arithmetic_t, NVAR> total_update{};

                        amr::containers::static_vector<arithmetic_t, NVAR> U_center;
                        auto fetch_center = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                            ((U_center[Is] = std::get<Is>(in_patches)[linear_idx]), ...);
                        };
                        fetch_center(std::make_index_sequence<NVAR>{});

                        for (int d = 0; d < DIM; ++d)
                        {
                            const auto stride = (d == 0) ? 1 : (d == 1 ? stride_y : stride_z);

                            amr::containers::static_vector<arithmetic_t, NVAR> U_L, U_R;
                            auto fetch_neighbors = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                                ((U_L[Is] = std::get<Is>(in_patches)[linear_idx - stride]), ...);
                                ((U_R[Is] = std::get<Is>(in_patches)[linear_idx + stride]), ...);
                            };
                            fetch_neighbors(std::make_index_sequence<NVAR>{});

                            amr::containers::static_vector<arithmetic_t, NVAR> fL, fR;

                            EquationT::rusanovFlux(U_L, U_center, fL, d, m_gamma);
                            EquationT::rusanovFlux(U_center, U_R, fR, d, m_gamma);

                            const arithmetic_t dt_over_dx = dt / c_size[d];

                            for (int k = 0; k < NVAR; ++k)
                            {
                                total_update[k] -= dt_over_dx * (fR[k] - fL[k]);
                            }
                        }

                        auto apply_update = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                            ((std::get<Is>(out_patches)[linear_idx] =
                                  std::get<Is>(in_patches)[linear_idx] + total_update[Is]),
                             ...);
                        };
                        apply_update(std::make_index_sequence<NVAR>{});
                    }
                );
            }
        );
        m_tree.swap_buffers();
        m_tree.halo_exchange_update();
    }

    auto compute_time_step_cpu() const -> arithmetic_t
    {
        // Global minimum time step, safely updated across threads
        // TODO: We could have a tighter upper bound here.
        //       maybe there is some other theoretical limit we can use instead
        std::atomic<arithmetic_t> dt{ std::numeric_limits<arithmetic_t>::max() };

        auto const r = std::views::iota(decltype(m_tree.size()){}, m_tree.size());
        std::for_each(
            AMR_EXECUTION_POLICY,
            std::cbegin(r),
            std::cend(r),
            [this, &dt](auto const p_idx) mutable
            {
                auto       local_dt{ std::numeric_limits<arithmetic_t>::max() };
                const auto patch_id = m_tree.get_node_index_at(p_idx);
                const auto c_size   = GeometryT::cell_sizes(patch_id);

                // Fetch the SoA patch tuple for this specific patch
                auto fetch_patches = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                    return std::forward_as_tuple(
                        m_tree.template get_patch<
                            typename std::tuple_element<Is, typename EquationT::FieldTags>::type
                        >(p_idx)...
                    );
                };
                auto patches = fetch_patches(std::make_index_sequence<NVAR>{});
                // Iterate over the cells
                amr::containers::manipulators::shaped_for<
                    typename patch_layout_t::interior_iteration_control_t>(
                    [this, &c_size, &patches](arithmetic_t& out_dt, auto const& idxs)
                    {
                        const auto linear_idx = padded_layout_t::linear_index(idxs);
                        // Ask Equation for max wave speed directly from the SoA tuple
                        // TODO: The level of abstraction here is incorrect in my
                        // opiniton. Iterating over the dimensions and calling it
                        // direction (in getMaxSpeed) is heavily missleading.
                        // Look at neighbor::direction as a suggested alternative.
                        for (int d = 0; d < DIM; ++d)
                        {
                            const arithmetic_t speed =
                                EquationT::getMaxSpeed(patches, linear_idx, d, m_gamma);
                            // TODO: This magic number should at least have a name
                            // TODO: Is this value special in any way?
                            if (speed > 1e-12)
                            {
                                out_dt = std::min(out_dt, c_size[d] / speed);
                            }
                        }
                    },
                    local_dt
                );
                auto current = dt.load();
                while (local_dt < current && !dt.compare_exchange_weak(current, local_dt))
                    ;
            }
        );
        return m_cfl * dt;
    }

#ifdef AMR_ENABLE_CUDA_AMR
    auto ensure_cuda_dt_resources() const -> void
    {
        if (m_device_dt_buffer == nullptr)
        {
            m_device_dt_buffer = static_cast<arithmetic_t*>(
                amr::cuda::device_malloc(sizeof(arithmetic_t))
            );
        }
        if (m_device_dt_accumulator_buffer == nullptr)
        {
            m_device_dt_accumulator_buffer = static_cast<arithmetic_t*>(
                amr::cuda::device_malloc(sizeof(arithmetic_t))
            );
        }
        if (m_device_remaining_time_buffer == nullptr)
        {
            m_device_remaining_time_buffer = static_cast<arithmetic_t*>(
                amr::cuda::device_malloc(sizeof(arithmetic_t))
            );
        }
        if (m_device_executed_step_count_buffer == nullptr)
        {
            m_device_executed_step_count_buffer = static_cast<std::uint32_t*>(
                amr::cuda::device_malloc(sizeof(std::uint32_t))
            );
        }
        if (m_pinned_dt_buffer == nullptr)
        {
            m_pinned_dt_buffer = static_cast<arithmetic_t*>(
                amr::cuda::host_pinned_malloc(sizeof(arithmetic_t))
            );
        }
        if (m_pinned_executed_step_count_buffer == nullptr)
        {
            m_pinned_executed_step_count_buffer = static_cast<std::uint32_t*>(
                amr::cuda::host_pinned_malloc(sizeof(std::uint32_t))
            );
        }
        if (m_dt_ready_fence == nullptr)
        {
            m_dt_ready_fence = amr::cuda::async_copy_fence_create();
        }
        if (m_dt_copy_fence == nullptr)
        {
            m_dt_copy_fence = amr::cuda::async_copy_fence_create();
        }
        if (m_dt_copy_stream == nullptr)
        {
            m_dt_copy_stream = amr::cuda::async_copy_stream_create();
        }
    }

    template <auto B>
    auto get_device_ro_ptrs(BufferTag<B> buffer_tag) const
        -> std::array<const double*, NVAR>
    {
        return [&]<std::size_t... Is>(std::index_sequence<Is...>)
        {
            return std::array<const double*, NVAR>{
                reinterpret_cast<const double*>(m_tree.template get_device_buffer<
                    typename std::tuple_element<Is, typename EquationT::FieldTags>::type,
                    decltype(buffer_tag)::value>())...
            };
        }(std::make_index_sequence<NVAR>{});
    }

    template <auto B>
    auto get_device_rw_ptrs(BufferTag<B> buffer_tag) -> std::array<double*, NVAR>
    {
        return [&]<std::size_t... Is>(std::index_sequence<Is...>)
        {
            return std::array<double*, NVAR>{
                reinterpret_cast<double*>(m_tree.template get_device_buffer<
                    typename std::tuple_element<Is, typename EquationT::FieldTags>::type,
                    decltype(buffer_tag)::value>())...
            };
        }(std::make_index_sequence<NVAR>{});
    }

    auto in_ptrs_as_mutable(std::array<const double*, NVAR> const& in_ptrs) const
        -> std::array<double*, NVAR>
    {
        std::array<double*, NVAR> mutable_ptrs{};
        for (std::size_t i = 0; i < NVAR; ++i)
        {
            mutable_ptrs[i] = const_cast<double*>(in_ptrs[i]);
        }
        return mutable_ptrs;
    }

    auto make_cuda_time_step_config() const -> amr::cuda::time_step_launch_config
    {
        amr::cuda::time_step_launch_config config{};
        config.num_patches     = m_tree.size();
        config.patch_flat_size = patch_layout_t::flat_size();
        config.data_flat_size  = patch_layout_t::data_layout_t::flat_size();
        config.halo_width      = patch_layout_t::halo_width();
        config.dt              = 0.0;
        config.cfl             = 0.0;
        config.gamma           = m_gamma;

        const auto root_size = GeometryT::cell_sizes(patch_index_t::root());
        config.root_c_size = { 0.0, 0.0, 0.0 };
        for (int d = 0; d < DIM; ++d)
        {
            config.root_c_size[d] = root_size[d];
            config.data_sizes[d]     = patch_layout_t::data_layout_t::sizes()[d];
            config.data_strides[d]   = patch_layout_t::data_layout_t::strides()[d];
            config.padded_strides[d] = patch_layout_t::padded_layout_t::strides()[d];
        }

        config.stride_y = patch_layout_t::padded_layout_t::shape_t::sizes()[DIM - 1];
        config.stride_z =
            (DIM == 3)
                ? patch_layout_t::padded_layout_t::shape_t::sizes()[1] * config.stride_y
                : 0;
        return config;
    }

    auto wait_for_pending_dt_copy() const -> void
    {
        if (m_dt_copy_pending != 0)
        {
            amr::cuda::async_copy_fence_wait(m_dt_copy_fence);
            m_dt_copy_pending = 0;
        }
    }
#endif
};

// Typedefs
template <typename TreeT, typename GeometryT, typename EquationT>
using amr_solver_2d = amr_solver<TreeT, GeometryT, EquationT, 2>;

template <typename TreeT, typename GeometryT, typename EquationT>
using amr_solver_3d = amr_solver<TreeT, GeometryT, EquationT, 3>;

#endif // AMR_SOLVER_HPP
