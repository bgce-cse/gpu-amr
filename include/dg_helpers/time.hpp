#pragma once

#include "../../build/generated_config.hpp"
#include "../containers/static_tensor.hpp"
#include <functional>
#include <memory>
#include <stdexcept>
#include <string_view>

namespace amr::time_integration
{

/**
 * @brief Abstract base class for time integrators (ODE solvers) on patch containers.
 */
template <typename PatchContainerT>
class Integrator
{
public:
    virtual ~Integrator() = default;

    /**
     * @brief Performs one time step.
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
};

/**
 * @brief Explicit Euler integrator for entire patches.
 */
template <typename PatchContainerT>
class ExplicitEuler : public Integrator<PatchContainerT>
{
private:
    std::remove_reference_t<PatchContainerT> patch_update;

public:
    ExplicitEuler()
        : patch_update{}
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
};

/**
 * @brief SSPRK2 integrator for entire patches.
 */
template <typename PatchContainerT>
class SSPRK2 : public Integrator<PatchContainerT>
{
private:
    std::remove_reference_t<PatchContainerT> stage1, stage2;

public:
    SSPRK2()
        : stage1{}
        , stage2{}
    {
    }

    void step(
        std::function<void(PatchContainerT&, const PatchContainerT&, double)> residual,
        PatchContainerT&                                                      patch_dofs,
        double                                                                time,
        double                                                                dt
    ) override
    {
        for (auto& elem : stage1)
            elem = {};
        for (auto& elem : stage2)
            elem = {};

        residual(stage1, patch_dofs, time);
        stage1 = patch_dofs + stage1 * dt;

        residual(stage2, stage1, time + dt);
        patch_dofs = (patch_dofs + stage1 + stage2 * dt) * 0.5;
    }
};

/**
 * @brief SSPRK3 integrator for entire patches.
 */
template <typename PatchContainerT>
class SSPRK3 : public Integrator<PatchContainerT>
{
private:
    std::remove_reference_t<PatchContainerT> stage1, stage2, stage3;

public:
    SSPRK3()
        : stage1{}
        , stage2{}
        , stage3{}
    {
    }

    void step(
        std::function<void(PatchContainerT&, const PatchContainerT&, double)> residual,
        PatchContainerT&                                                      patch_dofs,
        double                                                                time,
        double                                                                dt
    ) override
    {
        for (auto& elem : stage1)
            elem = {};
        for (auto& elem : stage2)
            elem = {};
        for (auto& elem : stage3)
            elem = {};

        residual(stage1, patch_dofs, time);
        stage1 = patch_dofs + stage1 * dt;

        residual(stage2, stage1, time + dt);
        stage2 = patch_dofs * 0.75 + (stage1 + stage2 * dt) * 0.25;

        residual(stage3, stage2, time + 0.5 * dt);
        patch_dofs = patch_dofs * (1.0 / 3.0) + (stage2 + stage3 * dt) * (2.0 / 3.0);
    }
};

/**
 * @brief Factory function to create a patch-based integrator from config.
 */
template <typename PatchContainerT>
inline std::shared_ptr<Integrator<PatchContainerT>>
    make_time_integrator(std::string_view integrator_name)
{
    if (integrator_name == "Euler" || integrator_name == "SSPRK1")
        return std::make_shared<ExplicitEuler<PatchContainerT>>();
    else if (integrator_name == "SSPRK2")
        return std::make_shared<SSPRK2<PatchContainerT>>();
    else if (integrator_name == "SSPRK3")
        return std::make_shared<SSPRK3<PatchContainerT>>();
    else
        throw std::invalid_argument(
            std::string("Unknown time integrator: ") + std::string(integrator_name)
        );
}

/**
 * @brief Convenience factory using config.
 */
template <typename PatchContainerT>
inline auto make_configured_time_integrator()
{
    return make_time_integrator<PatchContainerT>(amr::config::TimeIntegrator);
}

} // namespace amr::time_integration
