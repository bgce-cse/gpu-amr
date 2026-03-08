#pragma once

#include "containers/static_tensor.hpp"
#include "generated_config.hpp"
#include <string_view>

namespace amr::time_integration
{

/**
 * @brief Explicit Euler integrator for entire patches.
 *
 * All callable parameters are accepted as template arguments
 * so the compiler can inline the residual lambda directly.
 */
template <typename PatchContainerT>
class ExplicitEuler
{
private:
    std::remove_reference_t<PatchContainerT> patch_update;
    std::remove_reference_t<PatchContainerT> u_n; // u^n saved for stage interface

public:
    ExplicitEuler()
        : patch_update{}
        , u_n{}
    {
    }

    template <typename Residual>
    void step(Residual&& residual, PatchContainerT& patch_dofs, double time, double dt)
    {
        for (auto& elem : patch_update)
            elem = {};

        residual(patch_update, patch_dofs, time);
        patch_dofs += patch_update * dt;
    }

    static constexpr int num_stages()
    {
        return 1;
    }

    void begin_step(const PatchContainerT& patch_dofs)
    {
        u_n = patch_dofs;
    }

    template <typename Residual>
    void compute_stage(
        int /*stage*/,
        Residual&&             residual,
        const PatchContainerT& patch_dofs,
        double                 time,
        double                 dt
    )
    {
        for (auto& elem : patch_update)
            elem = {};
        residual(patch_update, patch_dofs, time);
        // Euler: u^{n+1} = u^n + dt * F(u^n)
        u_n = patch_dofs + patch_update * dt;
    }

    const PatchContainerT&
        stage_result(int /*stage*/, const PatchContainerT& /*patch_dofs*/)
    {
        return u_n;
    }

    void finish_step(PatchContainerT& patch_dofs)
    {
        patch_dofs = u_n;
    }
};

/**
 * @brief SSPRK2 integrator for entire patches.
 *
 *  Stage 0:  u* = u^n + dt * F(u^n)
 *  Stage 1:  u^{n+1} = 0.5 * (u^n + u* + dt * F(u*))
 */
template <typename PatchContainerT>
class SSPRK2
{
private:
    std::remove_reference_t<PatchContainerT> stage_rhs; // scratch for RHS eval
    std::remove_reference_t<PatchContainerT> u_n;       // u^n
    std::remove_reference_t<PatchContainerT> u_star;    // intermediate u*

public:
    SSPRK2()
        : stage_rhs{}
        , u_n{}
        , u_star{}
    {
    }

    template <typename Residual>
    void step(Residual&& residual, PatchContainerT& patch_dofs, double time, double dt)
    {
        for (auto& elem : stage_rhs)
            elem = {};

        residual(stage_rhs, patch_dofs, time);
        u_star = patch_dofs + stage_rhs * dt;

        for (auto& elem : stage_rhs)
            elem = {};
        residual(stage_rhs, u_star, time + dt);
        patch_dofs = (patch_dofs + u_star + stage_rhs * dt) * 0.5;
    }

    static constexpr int num_stages()
    {
        return 2;
    }

    void begin_step(const PatchContainerT& patch_dofs)
    {
        u_n = patch_dofs;
    }

    template <typename Residual>
    void compute_stage(
        int                    stage,
        Residual&&             residual,
        const PatchContainerT& patch_dofs,
        double                 time,
        double                 dt
    )
    {
        for (auto& elem : stage_rhs)
            elem = {};

        if (stage == 0)
        {
            // u* = u^n + dt * F(u^n)
            residual(stage_rhs, patch_dofs, time);
            u_star = patch_dofs + stage_rhs * dt;
        }
        else // stage == 1
        {
            // u^{n+1} = 0.5 * (u^n + u* + dt * F(u*))
            residual(stage_rhs, patch_dofs, time + dt);
            u_star = (u_n + patch_dofs + stage_rhs * dt) * 0.5;
        }
    }

    const PatchContainerT&
        stage_result(int /*stage*/, const PatchContainerT& /*patch_dofs*/)
    {
        return u_star;
    }

    void finish_step(PatchContainerT& patch_dofs)
    {
        patch_dofs = u_star;
    }
};

/**
 * @brief SSPRK3 integrator for entire patches.
 *
 *  Stage 0:  u^(1) = u^n + dt * F(u^n)
 *  Stage 1:  u^(2) = 3/4 u^n + 1/4 (u^(1) + dt * F(u^(1)))
 *  Stage 2:  u^{n+1} = 1/3 u^n + 2/3 (u^(2) + dt * F(u^(2)))
 */
template <typename PatchContainerT>
class SSPRK3
{
private:
    std::remove_reference_t<PatchContainerT> stage_rhs;
    std::remove_reference_t<PatchContainerT> u_n;
    std::remove_reference_t<PatchContainerT> u_stage; // current intermediate state

public:
    SSPRK3()
        : stage_rhs{}
        , u_n{}
        , u_stage{}
    {
    }

    template <typename Residual>
    void step(Residual&& residual, PatchContainerT& patch_dofs, double time, double dt)
    {
        for (auto& elem : stage_rhs)
            elem = {};
        residual(stage_rhs, patch_dofs, time);
        u_stage = patch_dofs + stage_rhs * dt;

        for (auto& elem : stage_rhs)
            elem = {};
        residual(stage_rhs, u_stage, time + dt);
        u_stage = patch_dofs * 0.75 + (u_stage + stage_rhs * dt) * 0.25;

        for (auto& elem : stage_rhs)
            elem = {};
        residual(stage_rhs, u_stage, time + 0.5 * dt);
        patch_dofs = patch_dofs * (1.0 / 3.0) + (u_stage + stage_rhs * dt) * (2.0 / 3.0);
    }

    static constexpr int num_stages()
    {
        return 3;
    }

    void begin_step(const PatchContainerT& patch_dofs)
    {
        u_n = patch_dofs;
    }

    template <typename Residual>
    void compute_stage(
        int                    stage,
        Residual&&             residual,
        const PatchContainerT& patch_dofs,
        double                 time,
        double                 dt
    )
    {
        for (auto& elem : stage_rhs)
            elem = {};

        if (stage == 0)
        {
            // u^(1) = u^n + dt * F(u^n)
            residual(stage_rhs, patch_dofs, time);
            u_stage = patch_dofs + stage_rhs * dt;
        }
        else if (stage == 1)
        {
            // u^(2) = 3/4 u^n + 1/4 (u^(1) + dt * F(u^(1)))
            residual(stage_rhs, patch_dofs, time + dt);
            u_stage = u_n * 0.75 + (patch_dofs + stage_rhs * dt) * 0.25;
        }
        else // stage == 2
        {
            // u^{n+1} = 1/3 u^n + 2/3 (u^(2) + dt * F(u^(2)))
            residual(stage_rhs, patch_dofs, time + 0.5 * dt);
            u_stage = u_n * (1.0 / 3.0) + (patch_dofs + stage_rhs * dt) * (2.0 / 3.0);
        }
    }

    const PatchContainerT&
        stage_result(int /*stage*/, const PatchContainerT& /*patch_dofs*/)
    {
        return u_stage;
    }

    void finish_step(PatchContainerT& patch_dofs)
    {
        patch_dofs = u_stage;
    }
};

} // namespace amr::time_integration
