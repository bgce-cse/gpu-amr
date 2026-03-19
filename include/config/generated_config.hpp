#ifndef AMR_INCLUDED_GENERATED_CONFIG
#define AMR_INCLUDED_GENERATED_CONFIG

/**
 * @file generated_config.hpp
 * @brief Auto-generated configuration constants from config.yaml
 * @warning Do not edit manually! Regenerated at build time.
 */

#include <cstddef>

namespace amr::config {

/// Equation type enumeration
enum class EquationType { Advection, Euler };

/// Time integration scheme enumeration
enum class TimeIntegratorType { Euler, SSPRK2, SSPRK3 };

/// Courant-Friedrichs-Lewy condition number
constexpr double CourantNumber = 0.5;

/// Global compile-time configuration policy
struct GlobalConfigPolicy
{
    static constexpr double EndTime = 1.0;
    static constexpr std::size_t Order = 2;
    static constexpr std::size_t Dim = 2;
    static constexpr std::size_t DOFs = 4;
    static constexpr std::size_t PatchSize = 4;
    static constexpr std::size_t HaloWidth = 1;
    static constexpr unsigned int MaxDepth = 6;
    static constexpr EquationType equation = EquationType::Euler;
    static constexpr TimeIntegratorType integrator = TimeIntegratorType::SSPRK2;
};

} // namespace amr::config

#endif // AMR_INCLUDED_GENERATED_CONFIG
