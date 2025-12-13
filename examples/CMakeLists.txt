cmake_minimum_required(VERSION 3.25)

project(GPU_AMR VERSION 0.1)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_BUILD_PARALLEL_LEVEL 8)

# Build type configurations
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

set(CXX_WARNINGS
	-fbounds-check
	-fvisibility=hidden
	-pedantic
	-Wall
	-Wconversion
	-Wdangling-else
	-Wdouble-promotion
	-Wduplicated-branches
	-Wduplicated-cond
	-Werror
	-Wextra
	-Wfloat-equal
	-Wformat
	-Winvalid-pch
	-Wlogical-op
	-Wmisleading-indentation
	-Wnull-dereference
	-Wodr
	-Wpointer-arith
	-Wrestrict
	-Wreturn-local-addr
	-Wshadow
	-Wswitch-default
	-Wswitch-enum
	-Wuninitialized
	-Wvla
)

set(CXX_DIAGNOSTICS
	-fconcepts-diagnostics-depth=3
	-fdiagnostics-color=always
	-fdiagnostics-path-format=inline-events
	-fdiagnostics-show-caret
	-fdiagnostics-show-template-tree
)

set(CXX_DEBUG_INFO
	-fno-omit-frame-pointer
	-fvar-tracking
	-fvar-tracking-assignments
	-ggdb3
	-gvariable-location-views
	-ginline-points
	-gstatement-frontiers
)

set(DEBUG_CXX_FLAGS
	-ffinite-math-only
	-fmax-errors=15
	-fno-eliminate-unused-debug-symbols
	-fno-inline
	-fno-default-inline
	-march=native
	
	-O0
)

set(RELEASE_CXX_FLAGS
	-fno-math-errno
	-fno-trapping-math
	-fstrength-reduce
	-march=native
	
	-O3
)


set(REL_DEB_INFO_CXX_FLAGS
	-fno-math-errno
	-fno-trapping-math
	-fstrength-reduce
	-march=native
	
	-O2
)

set(SANITIZER_FLAGS
	-fsanitize=address
	-fsanitize=bounds
	-fsanitize=float-cast-overflow
	-fsanitize=float-divide-by-zero
	-fsanitize=integer-divide-by-zero
	-fsanitize=leak
	-fsanitize=null
	-fsanitize=signed-integer-overflow
	-fsanitize=undefined
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CXX_FLAGS
      "${DEBUG_CXX_FLAGS}"
      "${CXX_DEBUG_INFO}"
      "${CXX_WARNINGS}"
      "${CXX_DIAGNOSTICS}"
  )
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(CXX_FLAGS
      "${RELEASE_CXX_FLAGS}"
      "${CXX_WARNINGS}"
      "${CXX_DIAGNOSTICS}"
  )
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
  set(CXX_FLAGS
      "${REL_DEB_INFO_CXX_FLAGS}"
      "${CXX_DEBUG_INFO}"
      "${CXX_WARNINGS}"
      "${CXX_DIAGNOSTICS}"
  )
else()
  message(ERROR "Not implemented build type: ${CMAKE_BUILD_TYPE}")
endif()

set(LINK_FLAGS "")

option(ENABLE_SANITIZERS "Enable compiler sanitizers for runtime checks" OFF)
if(ENABLE_SANITIZERS)
    list(APPEND CXX_FLAGS ${SANITIZER_FLAGS})
    list(APPEND LINK_FLAGS ${SANITIZER_FLAGS})
endif()

# Output directories
set(OUT_TAIL_DIR ${CMAKE_BUILD_TYPE})
set(OUTPUT_DIR ${CMAKE_BINARY_DIR}/bin/${OUT_TAIL_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR})

# ============================================================================
# YAML Config to C++ Header Preprocessing
# ============================================================================
set(CONFIG_YAML "${CMAKE_SOURCE_DIR}/../config.yaml")
set(GENERATED_CONFIG_HPP "${CMAKE_BINARY_DIR}/generated_config.hpp")

# Read the YAML file
if(EXISTS ${CONFIG_YAML})
    file(READ "${CONFIG_YAML}" CONFIG_TEXT)
    
    # Extract simulation parameters
    string(REGEX MATCH "end_time:[ ]*[0-9.]+" END_TIME_LINE "${CONFIG_TEXT}")
    string(REGEX MATCH "grid_elements:[ ]*[0-9]+" GRID_ELEMENTS_LINE "${CONFIG_TEXT}")
    string(REGEX MATCH "grid_size:[ ]*[0-9.]+" GRID_SIZE_LINE "${CONFIG_TEXT}")
    
    # Extract solver parameters
    string(REGEX MATCH "order:[ ]*[0-9]+" ORDER_LINE "${CONFIG_TEXT}")
    string(REGEX MATCH "courant:[ ]*[0-9.]+" COURANT_LINE "${CONFIG_TEXT}")
    string(REGEX MATCH "timeintegrator:[ ]*[A-Z0-9]+" TIMEINT_LINE "${CONFIG_TEXT}")
    
    # Extract equation and scenario
    string(REGEX MATCH "equation:[ ]*[a-z_]+" EQUATION_LINE "${CONFIG_TEXT}")
    string(REGEX MATCH "scenario:[ ]*[a-z_]+" SCENARIO_LINE "${CONFIG_TEXT}")
    
    # Extract Dim and DOFs (root level)
    string(REGEX MATCH "Dim:[ ]*[0-9]+" DIM_LINE "${CONFIG_TEXT}")
    string(REGEX MATCH "DOFs:[ ]*[0-9]+" DOFS_LINE "${CONFIG_TEXT}")
    string(REGEX MATCH "MaxDepth:[ ]*[0-9]+" MAXDEPTH_LINE "${CONFIG_TEXT}")
    
    # Parse simulation values
    string(REGEX REPLACE "end_time:[ ]*" "" END_TIME_VALUE "${END_TIME_LINE}")
    string(REGEX REPLACE "grid_elements:[ ]*" "" GRID_ELEMENTS_VALUE "${GRID_ELEMENTS_LINE}")
    string(REGEX REPLACE "grid_size:[ ]*" "" GRID_SIZE_VALUE "${GRID_SIZE_LINE}")
    
    # Parse solver values
    string(REGEX REPLACE "order:[ ]*" "" ORDER_VALUE "${ORDER_LINE}")
    string(REGEX REPLACE "courant:[ ]*" "" COURANT_VALUE "${COURANT_LINE}")
    string(REGEX REPLACE "timeintegrator:[ ]*" "" TIMEINT_VALUE "${TIMEINT_LINE}")
    
    # Parse equation values
    string(REGEX REPLACE "equation:[ ]*" "" EQUATION_VALUE "${EQUATION_LINE}")
    string(REGEX REPLACE "scenario:[ ]*" "" SCENARIO_VALUE "${SCENARIO_LINE}")
    
    # Parse Dim and DOFs
    string(REGEX REPLACE "Dim:[ ]*" "" DIM_VALUE "${DIM_LINE}")
    string(REGEX REPLACE "DOFs:[ ]*" "" DOFS_VALUE "${DOFS_LINE}")
    string(REGEX REPLACE "MaxDepth:[ ]*" "" MAXDEPTH_VALUE "${MAXDEPTH_LINE}")
    
    # Validate extracted values with defaults
    if(NOT ORDER_VALUE)
        set(ORDER_VALUE 2)
        message(WARNING "Order not found in config.yaml, using default: 2")
    endif()
    if(NOT DIM_VALUE)
        set(DIM_VALUE 2)
        message(WARNING "Dim not found in config.yaml, using default: 2")
    endif()
    if(NOT DOFS_VALUE)
        set(DOFS_VALUE 4)
        message(WARNING "DOFs not found in config.yaml, using default: 4")
    endif()
    if(NOT MAXDEPTH_VALUE)
        set(MAXDEPTH_VALUE 1)
        message(WARNING "MaxDepth not found in config.yaml, using default: 1")
    endif()
    if(NOT EQUATION_VALUE)
        set(EQUATION_VALUE "advection")
        message(WARNING "Equation not found in config.yaml, using default: advection")
    endif()
    if(NOT SCENARIO_VALUE)
        set(SCENARIO_VALUE "gaussian_wave")
        message(WARNING "Scenario not found in config.yaml, using default: gaussian_wave")
    endif()
    if(NOT END_TIME_VALUE)
        set(END_TIME_VALUE 4.0)
        message(WARNING "end_time not found in config.yaml, using default: 4.0")
    endif()
    if(NOT GRID_ELEMENTS_VALUE)
        set(GRID_ELEMENTS_VALUE 16)
        message(WARNING "grid_elements not found in config.yaml, using default: 16")
    endif()
    if(NOT GRID_SIZE_VALUE)
        set(GRID_SIZE_VALUE 1.0)
        message(WARNING "grid_size not found in config.yaml, using default: 1.0")
    endif()
    if(NOT COURANT_VALUE)
        set(COURANT_VALUE 0.5)
        message(WARNING "courant not found in config.yaml, using default: 0.5")
    endif()
    if(NOT TIMEINT_VALUE)
        set(TIMEINT_VALUE "SSPRK2")
        message(WARNING "timeintegrator not found in config.yaml, using default: SSPRK2")
    endif()
    
   # Determine enum value strings
if(EQUATION_VALUE STREQUAL "advection")
    set(EQUATION_ENUM "EquationType::Advection")
elseif(EQUATION_VALUE STREQUAL "euler")
    set(EQUATION_ENUM "EquationType::Euler")
else()
    message(FATAL_ERROR "Unknown equation: ${EQUATION_VALUE}")
endif()

# Determine time integrator enum value
if(TIMEINT_VALUE STREQUAL "Euler" OR TIMEINT_VALUE STREQUAL "SSPRK1")
    set(TIMEINT_ENUM "TimeIntegratorType::Euler")
elseif(TIMEINT_VALUE STREQUAL "SSPRK2")
    set(TIMEINT_ENUM "TimeIntegratorType::SSPRK2")
elseif(TIMEINT_VALUE STREQUAL "SSPRK3")
    set(TIMEINT_ENUM "TimeIntegratorType::SSPRK3")
else()
    message(FATAL_ERROR "Unknown time integrator: ${TIMEINT_VALUE}")
endif()

# Generate header
file(WRITE "${GENERATED_CONFIG_HPP}"
"#pragma once\n\n"
"/**\n"
" * @file generated_config.hpp\n"
" * @brief Auto-generated configuration constants from config.yaml\n"
" * @warning Do not edit manually! Regenerated at build time.\n"
" */\n\n"
"#include \"containers/static_shape.hpp\"\n"
"#include <string_view>\n"
"#include <cstddef>\n\n"
"namespace amr::config {\n\n"
"// Canonical size type: std::size_t\n"
"// Type aliases for container shapes\n"
"enum class EquationType { Advection, Euler };\n"
"enum class TimeIntegratorType { Euler, SSPRK2, SSPRK3 };\n"
"constexpr double CourantNumber = ${COURANT_VALUE};\n"
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
"} // namespace amr::config\n"
)

    
    message(STATUS "Generated config header: ${GENERATED_CONFIG_HPP}")
    message(STATUS "  === Compile-time DG Configuration ===")
    message(STATUS "  Order      = ${ORDER_VALUE}")
    message(STATUS "  Dim        = ${DIM_VALUE}")
    message(STATUS "  DOFs       = ${DOFS_VALUE}")
    message(STATUS "  Equation   = ${EQUATION_VALUE}")
    message(STATUS "  Scenario   = ${SCENARIO_VALUE}")
    message(STATUS "  === Simulation Parameters ===")
    message(STATUS "  EndTime    = ${END_TIME_VALUE}")
    message(STATUS "  GridElements = ${GRID_ELEMENTS_VALUE}")
    message(STATUS "  GridSize   = ${GRID_SIZE_VALUE}")
    message(STATUS "  === Solver Parameters ===")
    message(STATUS "  Courant    = ${COURANT_VALUE}")
    message(STATUS "  TimeIntegrator = ${TIMEINT_VALUE}")
else()
    message(FATAL_ERROR "config.yaml not found at ${CONFIG_YAML}")
endif()

set(EXAMPLES_DIR "./")
find_package(TBB REQUIRED)
find_package(VTK REQUIRED COMPONENTS
    CommonCore
    IOXML
    IOLegacy
    ImagingCore
    CommonDataModel
)

# Generate executable for all .e.cpp
file(GLOB EXAMPLE_FILES "${EXAMPLES_DIR}/*.e.cpp")
foreach(EXAMPLE_FILE ${EXAMPLE_FILES})
    get_filename_component(TARGET_NAME ${EXAMPLE_FILE} NAME_WE)
    add_executable(${TARGET_NAME} ${EXAMPLE_FILE})
    target_include_directories(${TARGET_NAME} PUBLIC ../include ../lib ${CMAKE_BINARY_DIR})
    target_compile_options(${TARGET_NAME} PRIVATE ${CXX_FLAGS})
    target_link_options(${TARGET_NAME} PRIVATE ${LINK_FLAGS})
    target_link_libraries(${TARGET_NAME} PRIVATE TBB::tbb ${VTK_LIBRARIES})
    target_include_directories(${TARGET_NAME} PRIVATE ${VTK_INCLUDE_DIRS})
    # Ensure generated config is created before compiling
    add_dependencies(${TARGET_NAME} generated_config_header)
endforeach()

# Custom target to ensure header generation
add_custom_target(generated_config_header
    DEPENDS ${GENERATED_CONFIG_HPP}
)



# Build messages
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Output directory: ${OUTPUT_DIR}")
message(STATUS "Sanitizers: ${ENABLE_SANITIZERS}")