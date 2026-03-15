# ============================================================================
# Configuration Parsing Macro
# Extracts value from YAML line, provides default if not found
# ============================================================================
macro(parse_config_value VAR_NAME REGEX_LINE DEFAULT_VALUE)
    string(REGEX REPLACE "^[^:]+:[ ]*" "" ${VAR_NAME} "${${REGEX_LINE}}")
    if(NOT ${VAR_NAME})
        set(${VAR_NAME} ${DEFAULT_VALUE})
        message(STATUS "  ⚠️  ${VAR_NAME} not found, using default: ${DEFAULT_VALUE}")
    else()
        message(STATUS "  ✓ ${VAR_NAME}: ${${VAR_NAME}}")
    endif()
endmacro()

function(generate_config_header YAML_FILE OUTPUT_HPP)
    file(READ "${YAML_FILE}" CONFIG_TEXT)

    # Simulation parameters
    string(REGEX MATCH "end_time:[ ]*[0-9.]+" END_TIME_LINE "${CONFIG_TEXT}")
    string(REGEX MATCH "grid_elements:[ ]*[0-9]+" GRID_ELEMENTS_LINE "${CONFIG_TEXT}")
    string(REGEX MATCH "grid_size:[ ]*[0-9.]+" GRID_SIZE_LINE "${CONFIG_TEXT}")

    # Solver parameters
    string(REGEX MATCH "order:[ ]*[0-9]+" ORDER_LINE "${CONFIG_TEXT}")
    string(REGEX MATCH "courant:[ ]*[0-9.]+" COURANT_LINE "${CONFIG_TEXT}")
    string(REGEX MATCH "timeintegrator:[ ]*[A-Z0-9]+" TIMEINT_LINE "${CONFIG_TEXT}")

    # Equation and scenario
    string(REGEX MATCH "equation:[ ]*[a-z_]+" EQUATION_LINE "${CONFIG_TEXT}")
    string(REGEX MATCH "scenario:[ ]*[a-z_]+" SCENARIO_LINE "${CONFIG_TEXT}")

    # Domain and discretization
    string(REGEX MATCH "Dim:[ ]*[0-9]+" DIM_LINE "${CONFIG_TEXT}")
    string(REGEX MATCH "DOFs:[ ]*[0-9]+" DOFS_LINE "${CONFIG_TEXT}")
    string(REGEX MATCH "MaxDepth:[ ]*[0-9]+" MAXDEPTH_LINE "${CONFIG_TEXT}")

    message(STATUS "Configuration values:")
    parse_config_value(END_TIME_VALUE END_TIME_LINE "4.0")
    parse_config_value(GRID_ELEMENTS_VALUE GRID_ELEMENTS_LINE "16")
    parse_config_value(GRID_SIZE_VALUE GRID_SIZE_LINE "1.0")
    parse_config_value(ORDER_VALUE ORDER_LINE "2")
    parse_config_value(COURANT_VALUE COURANT_LINE "0.5")
    parse_config_value(TIMEINT_VALUE TIMEINT_LINE "SSPRK2")
    parse_config_value(EQUATION_VALUE EQUATION_LINE "advection")
    parse_config_value(SCENARIO_VALUE SCENARIO_LINE "gaussian_wave")
    parse_config_value(DIM_VALUE DIM_LINE "2")
    parse_config_value(DOFS_VALUE DOFS_LINE "4")
    parse_config_value(MAXDEPTH_VALUE MAXDEPTH_LINE "1")

    # ============================================================================
    # Map Configuration Values to C++ Enums
    # ============================================================================
    # Equation type enum
    if(EQUATION_VALUE STREQUAL "advection")
        set(EQUATION_ENUM "EquationType::Advection")
    elseif(EQUATION_VALUE STREQUAL "euler")
        set(EQUATION_ENUM "EquationType::Euler")
    else()
        message(FATAL_ERROR "Unknown equation type: ${EQUATION_VALUE}")
    endif()

    # Time integrator enum
    if(TIMEINT_VALUE STREQUAL "Euler" OR TIMEINT_VALUE STREQUAL "SSPRK1")
        set(TIMEINT_ENUM "TimeIntegratorType::Euler")
    elseif(TIMEINT_VALUE STREQUAL "SSPRK2")
        set(TIMEINT_ENUM "TimeIntegratorType::SSPRK2")
    elseif(TIMEINT_VALUE STREQUAL "SSPRK3")
        set(TIMEINT_ENUM "TimeIntegratorType::SSPRK3")
    else()
        message(FATAL_ERROR "Unknown time integrator: ${TIMEINT_VALUE}")
    endif()

    # Generate configuration header file
    file(WRITE "${OUTPUT_HPP}"
"#ifndef AMR_INCLUDED_GENERATED_CONFIG\n"
"#define AMR_INCLUDED_GENERATED_CONFIG\n\n"
"/**\n"
" * @file generated_config.hpp\n"
" * @brief Auto-generated configuration constants from config.yaml\n"
" * @warning Do not edit manually! Regenerated at build time.\n"
" */\n\n"
"#include <cstddef>\n\n"
"namespace amr::config {\n\n"
"/// Equation type enumeration\n"
"enum class EquationType { Advection, Euler };\n\n"
"/// Time integration scheme enumeration\n"
"enum class TimeIntegratorType { Euler, SSPRK2, SSPRK3 };\n\n"
"/// Courant-Friedrichs-Lewy condition number\n"
"constexpr double CourantNumber = ${COURANT_VALUE};\n\n"
"/// Global compile-time configuration policy\n"
"struct GlobalConfigPolicy\n"
"{\n"
"    static constexpr double EndTime = ${END_TIME_VALUE};\n"
"    static constexpr std::size_t Order = ${ORDER_VALUE};\n"
"    static constexpr std::size_t Dim = ${DIM_VALUE};\n"
"    static constexpr std::size_t DOFs = ${DOFS_VALUE};\n"
"    static constexpr std::size_t PatchSize = ${GRID_ELEMENTS_VALUE};\n"
"    static constexpr std::size_t HaloWidth = 1;\n"
"    static constexpr unsigned int MaxDepth = ${MAXDEPTH_VALUE};\n"
"    static constexpr EquationType equation = ${EQUATION_ENUM};\n"
"    static constexpr TimeIntegratorType integrator = ${TIMEINT_ENUM};\n"
"};\n\n"
"} // namespace amr::config\n\n"
"#endif // AMR_INCLUDED_GENERATED_CONFIG\n"
    )

    add_custom_target(generated_config_header DEPENDS ${OUTPUT_HPP})
endfunction()
