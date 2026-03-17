cmake_minimum_required(VERSION 3.25)
project(gpu_amr VERSION 0.1 LANGUAGES CXX)

# ----------------------------
# Global settings
# ----------------------------
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_CONFIGURATION_TYPES)
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
    endif()
endif()

# ----------------------------
# Options
# ----------------------------
option(ENABLE_SANITIZERS "Enable runtime sanitizers (Debug only)" ON)

# ----------------------------
# Execution policy
# ----------------------------
set(EXECUTION "SEQ" CACHE STRING "Execution policy")

set_property(CACHE EXECUTION PROPERTY STRINGS SEQ PAR)

# ----------------------------
# Output dir
# ----------------------------
set(OUTPUT_DIR_EXAMPLES   ${CMAKE_BINARY_DIR}/bin/examples)
set(OUTPUT_DIR_BENCHMARKS ${CMAKE_BINARY_DIR}/bin/benchmarks)

# ----------------------------
# Dependencies (global)
# ----------------------------
find_package(TBB REQUIRED)

# ----------------------------
# Logging
# ----------------------------
if(NOT TARGET spdlog)
    # Stand-alone build
    find_package(spdlog REQUIRED)
endif()

add_library(utility_logging
    src/utility/logging.cpp
)

target_include_directories(utility_logging
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(utility_logging
    PUBLIC spdlog::spdlog
)

target_compile_definitions(utility_logging PUBLIC
    # SPDLOG_USE_STD_FORMAT
    LOGGING_BUILD_TYPE="$<CONFIG>"
    LOGGING_EXECUTION="${EXECUTION}"
    LOGGING_GIT_HASH="${GIT_HASH}"
    LOGGING_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}"
    LOGGING_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}"
)

# ----------------------------
# Logging
# ----------------------------
set(LOG_LEVEL "INFO" CACHE STRING "Logging level")

set_property(CACHE LOG_LEVEL PROPERTY STRINGS
    TRACE DEBUG INFO WARNING ERROR FATAL OFF)

# ----------------------------
# Contracts
# ----------------------------
set(CONTRACTS ON CACHE BOOL "Enable contracts")

set_property(CACHE CONTRACTS PROPERTY VALUE ON OFF)

# ----------------------------
# Options
# ----------------------------
include(cmake/amr_compiler_options.cmake)

# ----------------------------
# Print Status
# ----------------------------
message(STATUS "----------------------------")
message(STATUS "gpu_amr configuration summary:")
message(STATUS "  Build type       : ${CMAKE_BUILD_TYPE}")
message(STATUS "  Execution policy : ${EXECUTION}")
message(STATUS "  Log level        : ${LOG_LEVEL}")
message(STATUS "  Sanitizers       : ${ENABLE_SANITIZERS}")
message(STATUS "  Output dir       : ${OUTPUT_DIR_EXAMPLES}")
message(STATUS "----------------------------")

# ----------------------------
# Core taget
# ----------------------------
add_library(gpu_amr INTERFACE)

target_compile_features(gpu_amr INTERFACE cxx_std_23)

target_include_directories(gpu_amr INTERFACE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/lib
)

target_link_libraries(gpu_amr
    INTERFACE TBB::tbb
    INTERFACE utility_logging
    INTERFACE amr_compiler_options
)

# ----------------------------
# Compiler flags
# ----------------------------
if(NOT DEFINED AMR_WARNINGS)
    message(FATAL_ERROR "Compiler warning flags not provided. Use a toolchain file.")
endif()
if(NOT DEFINED AMR_DIAGNOSTICS)
    message(FATAL_ERROR "Compiler diagnostic flags not provided. Use a toolchain file.")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ----------------------------
# Generated Config
# ----------------------------

include(cmake/functions/GenerateConfig.cmake)
generate_config_header(
    "${PROJECT_SOURCE_DIR}/config/config.yaml"
    "${PROJECT_SOURCE_DIR}/include/config/generated_config.hpp"
)

# ----------------------------
# Policy targets
# ----------------------------
add_library(amr_examples INTERFACE)
add_library(amr_benchmarks INTERFACE)
add_library(amr_tests INTERFACE)

target_compile_definitions(amr_examples INTERFACE
    $<$<BOOL:${CONTRACTS}>:AMR_ENABLE_CONTRACTS=1>
    AMR_LOG_LEVEL=${LOG_LEVEL}
    AMR_EXECUTION=${EXECUTION}
)

target_compile_definitions(amr_benchmarks INTERFACE
    AMR_LOG_LEVEL=OFF
    AMR_EXECUTION=${EXECUTION}
)

target_compile_definitions(amr_tests INTERFACE
    $<$<BOOL:${CONTRACTS}>:AMR_ENABLE_CONTRACTS=1>
    AMR_LOG_LEVEL=OFF
    AMR_EXECUTION=${EXECUTION}
)

# ----------------------------
# CMake Subdirectories
# ----------------------------

add_subdirectory(examples)
add_subdirectory(tests)
add_subdirectory(benchmark)
