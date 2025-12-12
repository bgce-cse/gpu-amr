#ifndef DG_HELPERS_TIME_INTEGRATION_HPP
#define DG_HELPERS_TIME_INTEGRATION_HPP

#include "generated_config.hpp"
#include "time.hpp"

namespace amr::time_integration
{
/**
 * @brief Traits template for time integrator selection (C++20)
 *
 * Allows compile-time selection of time integrator based on TimeIntegratorType enum.
 * Usage: typename TimeIntegratorTraits<amr::config::TimeIntegratorType::SSPRK2,
 * PatchContainerT>::type
 */
template <amr::config::TimeIntegratorType IntegratorType, typename PatchContainerT>
struct TimeIntegratorTraits;

template <typename PatchContainerT>
struct TimeIntegratorTraits<amr::config::TimeIntegratorType::Euler, PatchContainerT>
{
    using type = ExplicitEuler<PatchContainerT>;
};

template <typename PatchContainerT>
struct TimeIntegratorTraits<amr::config::TimeIntegratorType::SSPRK2, PatchContainerT>
{
    using type = SSPRK2<PatchContainerT>;
};

template <typename PatchContainerT>
struct TimeIntegratorTraits<amr::config::TimeIntegratorType::SSPRK3, PatchContainerT>
{
    using type = SSPRK3<PatchContainerT>;
};
} // namespace amr::time_integration

#endif // DG_HELPERS_TIME_INTEGRATION_HPP
