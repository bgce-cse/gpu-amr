#pragma once

#include "containers/static_tensor.hpp"
#include "generated_config.hpp"
#include <functional>
#include <memory>
#include <stdexcept>
#include <string_view>

namespace amr::time_integration
{

/**
 * @brief Abstract base class for time integrators (ODE solvers) on patch containers.
 *
 * Provides both the legacy single-call step() and a stage-wise interface
 * (num_stages / begin_step / compute_stage / finish_step) that allows the
 * solver to perform a global halo exchange between stages.
 */
template <typename PatchContainerT>
class Integrator
{
public:
    virtual ~Integrator() = default;

    /**
     * @brief Performs one complete time step (all stages at once).
     *
     * @warning This method evaluates intermediate stages using LOCAL buffers
     * whose halo regions are stale (they don't reflect the intermediate state
     * of neighboring patches).  For multi-stage methods this is INCORRECT
     * when inter-patch coupling exists (e.g. the DG surface flux).
     * Prefer the stage-wise interface in the solver loop instead.
     *
     * @param residual Function computing du/dt from current patch
     * @param patch_dofs Current patch DOFs (modified in-place)
     * @param time Current simulation time
     * @param dt Time step size
     */
    virtual void step(
        std::function<void(PatchContainerT&, const PatchContainerT&, double)> residual,
        PatchContainerT&                                                      patch_dofs,
        double                                                                time,
        double                                                                dt
    ) = 0;

    // -----------------------------------------------------------------
    //  Stage-wise interface
    //
    //  Usage from the solver:
    //    integrator.begin_step(patch_dofs);
    //    for (int s = 0; s < integrator.num_stages(); ++s) {
    //        // halo exchange (so RHS sees consistent data)
    //        for each patch:
    //            integrator.compute_stage(s, residual, patch_dofs, time, dt);
    //        // write the intermediate state back into the tree for halo exch.
    //        for each patch:
    //            patch_dofs = integrator.stage_result(s, patch_dofs);
    //    }
    //    for each patch:
    //        integrator.finish_step(patch_dofs);
    // -----------------------------------------------------------------

    /** Number of RHS evaluations (stages) per time step. */
    virtual int num_stages() const = 0;

    /** Store u^n before the first stage. */
    virtual void begin_step(const PatchContainerT& patch_dofs) = 0;

    /**
     * Evaluate the RHS for the given stage and accumulate into internal storage.
     * @param stage  Stage index, 0-based.
     * @param patch_dofs  The CURRENT state of this patch (should include valid halos).
     */
    virtual void compute_stage(
        int                                                                   stage,
        std::function<void(PatchContainerT&, const PatchContainerT&, double)> residual,
        const PatchContainerT&                                                patch_dofs,
        double                                                                time,
        double                                                                dt
    ) = 0;

    /**
     * Return the intermediate state that should be written back to the tree
     * after stage `stage` so halo exchange can see it.
     */
    virtual PatchContainerT
        stage_result(int stage, const PatchContainerT& patch_dofs) = 0;

    /**
     * Combine all stage data and write the final u^{n+1} into patch_dofs.
     */
    virtual void finish_step(PatchContainerT& patch_dofs) = 0;
};

/**
 * @brief Explicit Euler integrator for entire patches.
 */
template <typename PatchContainerT>
class ExplicitEuler : public Integrator<PatchContainerT>
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

    void step(
        std::function<void(PatchContainerT&, const PatchContainerT&, double)> residual,
        PatchContainerT&                                                      patch_dofs,
        double                                                                time,
        double                                                                dt
    ) override
    {
        // zero the update
        for (auto& elem : patch_update)
            elem = {};

        residual(patch_update, patch_dofs, time);
        patch_dofs = patch_dofs + (patch_update * dt);
    }

    int num_stages() const override
    {
        return 1;
    }

    void begin_step(const PatchContainerT& patch_dofs) override
    {
        u_n = patch_dofs;
    }

    void compute_stage(
        int /*stage*/,
        std::function<void(PatchContainerT&, const PatchContainerT&, double)> residual,
        const PatchContainerT&                                                patch_dofs,
        double                                                                time,
        double                                                                dt
    ) override
    {
        for (auto& elem : patch_update)
            elem = {};
        residual(patch_update, patch_dofs, time);
        // Euler: u^{n+1} = u^n + dt * F(u^n)
        u_n = patch_dofs + patch_update * dt;
    }

    PatchContainerT
        stage_result(int /*stage*/, const PatchContainerT& /*patch_dofs*/) override
    {
        return u_n;
    }

    void finish_step(PatchContainerT& patch_dofs) override
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
class SSPRK2 : public Integrator<PatchContainerT>
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

    void step(
        std::function<void(PatchContainerT&, const PatchContainerT&, double)> residual,
        PatchContainerT&                                                      patch_dofs,
        double                                                                time,
        double                                                                dt
    ) override
    {
        // Legacy all-in-one (stale-halo issue with inter-patch coupling)
        for (auto& elem : stage_rhs)
            elem = {};

        residual(stage_rhs, patch_dofs, time);
        u_star = patch_dofs + stage_rhs * dt;

        for (auto& elem : stage_rhs)
            elem = {};
        residual(stage_rhs, u_star, time + dt);
        patch_dofs = (patch_dofs + u_star + stage_rhs * dt) * 0.5;
    }

    int num_stages() const override
    {
        return 2;
    }

    void begin_step(const PatchContainerT& patch_dofs) override
    {
        u_n = patch_dofs;
    }

    void compute_stage(
        int                                                                   stage,
        std::function<void(PatchContainerT&, const PatchContainerT&, double)> residual,
        const PatchContainerT&                                                patch_dofs,
        double                                                                time,
        double                                                                dt
    ) override
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
            // patch_dofs is now the halo-exchanged u*
            residual(stage_rhs, patch_dofs, time + dt);
            // u^{n+1} = 0.5 * (u^n + u* + dt * F(u*))
            u_star = (u_n + patch_dofs + stage_rhs * dt) * 0.5;
        }
    }

    PatchContainerT
        stage_result(int /*stage*/, const PatchContainerT& /*patch_dofs*/) override
    {
        return u_star;
    }

    void finish_step(PatchContainerT& patch_dofs) override
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
class SSPRK3 : public Integrator<PatchContainerT>
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

    void step(
        std::function<void(PatchContainerT&, const PatchContainerT&, double)> residual,
        PatchContainerT&                                                      patch_dofs,
        double                                                                time,
        double                                                                dt
    ) override
    {
        // Legacy all-in-one (stale-halo issue with inter-patch coupling)
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

    int num_stages() const override
    {
        return 3;
    }

    void begin_step(const PatchContainerT& patch_dofs) override
    {
        u_n = patch_dofs;
    }

    void compute_stage(
        int                                                                   stage,
        std::function<void(PatchContainerT&, const PatchContainerT&, double)> residual,
        const PatchContainerT&                                                patch_dofs,
        double                                                                time,
        double                                                                dt
    ) override
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
            // patch_dofs is now halo-exchanged u^(1)
            // u^(2) = 3/4 u^n + 1/4 (u^(1) + dt * F(u^(1)))
            residual(stage_rhs, patch_dofs, time + dt);
            u_stage = u_n * 0.75 + (patch_dofs + stage_rhs * dt) * 0.25;
        }
        else // stage == 2
        {
            // patch_dofs is now halo-exchanged u^(2)
            // u^{n+1} = 1/3 u^n + 2/3 (u^(2) + dt * F(u^(2)))
            residual(stage_rhs, patch_dofs, time + 0.5 * dt);
            u_stage = u_n * (1.0 / 3.0) + (patch_dofs + stage_rhs * dt) * (2.0 / 3.0);
        }
    }

    PatchContainerT
        stage_result(int /*stage*/, const PatchContainerT& /*patch_dofs*/) override
    {
        return u_stage;
    }

    void finish_step(PatchContainerT& patch_dofs) override
    {
        patch_dofs = u_stage;
    }
};

} // namespace amr::time_integration
