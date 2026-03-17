#ifndef DG_HELPERS_SCENARIO_HPP
#define DG_HELPERS_SCENARIO_HPP

#include <stdexcept>
#include <string>

namespace amr::equations
{

/**
 * @brief Scenario enumeration for different physical scenarios
 *
 * Defines different physical scenarios (e.g., planar waves, shock tube, Gaussian bump).
 */
enum class Scenario
{
    PlanarWaves,
    GaussianWave,
    ShockTube
};

/**
 * @brief Check if boundary conditions should be periodic for a given scenario
 *
 * @param scenario Scenario enum value
 * @return true if all boundaries are periodic
 */
constexpr bool is_periodic_boundary(Scenario scenario)
{
    return scenario == Scenario::PlanarWaves;
}

/**
 * @brief Factory function to create scenario from string name
 *
 * Supported: "planar_waves", "gaussian_wave", "shock_tube"
 */
inline Scenario make_scenario(const std::string& scenario_name)
{
    if (scenario_name == "planar_waves")
    {
        return Scenario::PlanarWaves;
    }
    else if (scenario_name == "gaussian_wave")
    {
        return Scenario::GaussianWave;
    }
    else if (scenario_name == "shock_tube")
    {
        return Scenario::ShockTube;
    }
    else
    {
        throw std::invalid_argument("Unknown scenario name: " + scenario_name);
    }
}

/**
 * @brief Convert scenario enum to string representation
 *
 * @param scenario Scenario enum value
 * @return String representation of the scenario
 */
inline const char* scenario_to_string(Scenario scenario)
{
    switch (scenario)
    {
        case Scenario::PlanarWaves: return "planar_waves";
        case Scenario::GaussianWave: return "gaussian_wave";
        case Scenario::ShockTube: return "shock_tube";
        default: return "unknown";
    }
}

} // namespace amr::equations

#endif // DG_HELPERS_SCENARIO_HPP
