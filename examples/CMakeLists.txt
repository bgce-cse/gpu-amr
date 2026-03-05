cmake_minimum_required(VERSION 3.25)

project(GPU_AMR VERSION 0.1 LANGUAGES CXX)

# ============================================================================
# C++ Standard Configuration
# ============================================================================
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================================
# Build System Configuration
# ============================================================================
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_BUILD_PARALLEL_LEVEL 8)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default to Release build if not specified (production-ready default)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    message(STATUS "⚙️  No build type specified. Defaulting to Release (production-ready)")
endif()

# ============================================================================
# Common Compiler Flags
# Applied to all build types for consistency
# ============================================================================

# Warning flags - strict error checking
set(CXX_WARNINGS
    -Wall
    -Wextra
    -Werror
    -pedantic
    -Wconversion
    -Wshadow
    -Wnull-dereference
    -Wdouble-promotion
    -Wformat=2
    -Wfloat-equal
    -Wpointer-arith
    -Wswitch-default
    -Wswitch-enum
    -Wuninitialized
    -Wmisleading-indentation
    -Wduplicated-cond
    -Wduplicated-branches
    -Wlogical-op
    -Wdangling-else
    -Wrestrict
    -Wreturn-local-addr
    -Winvalid-pch
    -Wodr
    -Wvla
    -fbounds-check
    -fvisibility=hidden
)

# Compiler diagnostics - enhanced error messages
set(CXX_DIAGNOSTICS
    -fdiagnostics-color=always
    -fdiagnostics-show-caret
    -fdiagnostics-show-template-tree
)

# ============================================================================
# Debug Build: Full debugging support, no optimization
# Ideal for development and debugging
# ============================================================================
set(DEBUG_FLAGS
    # Optimization (disabled for better debugging)
    -O0
    -fno-inline
    
    # Debug symbols - comprehensive information
    -g3                              # Maximum debug info
    -ggdb3                           # GDB-specific extensions
    -fno-omit-frame-pointer          # Keep frame pointers for profiling
    -fvar-tracking                   # Track variable locations
    -fvar-tracking-assignments       # Track variable assignments
    -gvariable-location-views        # Generate location view info
    -ginline-points                  # Include inline points
    -gstatement-frontiers            # Statement frontier debugging
    -fno-eliminate-unused-debug-symbols
    
    # Architecture
    -march=native
)

# ============================================================================
# Release Build: Maximum optimization, production-ready
# Ideal for performance-critical deployments
# ============================================================================
set(RELEASE_FLAGS
    # Optimization (maximum performance)
    -O3                              # Maximum optimization level
    -march=native                    # Native architecture tuning
    -fstrength-reduce                # Replace multiply with add
    
    # Math optimizations (fast math)
    -fno-math-errno                  # Assume math functions won't set errno
    -fno-trapping-math               # Assume no math traps
    -ffast-math                      # Aggressive floating-point optimizations
    
    # Advanced optimizations
    -flto                            # Link-time optimization
    -fomit-frame-pointer             # Omit frame pointers (saves registers)
)

# ============================================================================
# RelWithDebInfo Build: Balanced - optimization with debug symbols
# Ideal for profiling and debugging optimized code
# ============================================================================
set(RELWITHDEBINFO_FLAGS
    # Optimization (balanced)
    -O2                              # Good balance of speed and debug info
    -march=native                    # Native architecture tuning
    -fstrength-reduce                # Replace multiply with add
    
    # Debug information (for profiling)
    -g3                              # Comprehensive debug info
    -ggdb3                           # GDB extensions
    -fno-omit-frame-pointer          # Keep frame pointers for profiling
    -gvariable-location-views        # Generate location view info
    
    # Math optimizations
    -fno-math-errno                  # Assume math functions won't set errno
    -fno-trapping-math               # Assume no math traps
)

# ============================================================================
# Sanitizer Configuration (Optional, for testing)
# Enables runtime checks for common bugs
# Adds significant overhead (~2-3x slowdown) but catches bugs early
# ============================================================================
set(SANITIZER_FLAGS
    -fsanitize=address               # Memory access bugs
    -fsanitize=leak                  # Memory leaks
    -fsanitize=undefined             # Undefined behavior
    -fsanitize=bounds                # Array bounds violations
    -fsanitize=null                  # Null pointer dereferences
    -fsanitize=integer-divide-by-zero # Division by zero
    -fsanitize=float-divide-by-zero  # Float division by zero
    -fsanitize=float-cast-overflow   # Float cast overflow
    -fsanitize=signed-integer-overflow # Signed integer overflow
)

option(ENABLE_SANITIZERS "Enable compiler sanitizers for runtime checks (adds ~2-3x overhead)" OFF)

# ============================================================================
# Apply Compiler Flags Based on Build Type and Generator
# ============================================================================

message(STATUS "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
message(STATUS "Build Configuration Summary")
message(STATUS "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
message(STATUS "Build Type:      ${CMAKE_BUILD_TYPE}")
message(STATUS "Generator:       ${CMAKE_GENERATOR}")
message(STATUS "Compiler:        ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ Standard:    C++${CMAKE_CXX_STANDARD}")

if(CMAKE_GENERATOR STREQUAL "Xcode")
    message(STATUS "")
    message(STATUS "⚠️  Xcode Generator Detected - Using AppleClang-Compatible Flags")
    message(STATUS "    (Full compiler flags require Unix Makefiles generator)")
    message(STATUS "")
    
    # Xcode with AppleClang - simplified flag set
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CXX_FLAGS -g -O0 -Wall -Wextra)
        message(STATUS "Debug Mode:      ✓ Optimizations disabled, full debug symbols")
    elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
        set(CXX_FLAGS -O2 -g -Wall -Wextra -fvisibility=default)
        message(STATUS "RelWithDebInfo:  ✓ Balanced optimization with debug symbols")
    elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
        set(CXX_FLAGS -O3 -Wall -Wextra)
        message(STATUS "Release Mode:    ✓ Full optimization, production-ready")
    else()
        message(FATAL_ERROR "Unsupported build type for Xcode: ${CMAKE_BUILD_TYPE}")
    endif()
else()
    # Unix Makefiles and other generators - full feature flags
    message(STATUS "")
    message(STATUS "✓ Full Compiler Flags Configuration Enabled")
    message(STATUS "")
    
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CXX_FLAGS
            ${DEBUG_FLAGS}
            ${CXX_WARNINGS}
            ${CXX_DIAGNOSTICS}
        )
        message(STATUS "Debug Mode:      ✓ O0, Full debug symbols, frame pointers")
        message(STATUS "                 ✓ Variable tracking enabled")
        message(STATUS "                 ✓ Better debugger experience")
        
    elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
        set(CXX_FLAGS
            ${RELEASE_FLAGS}
            ${CXX_WARNINGS}
            ${CXX_DIAGNOSTICS}
        )
        message(STATUS "Release Mode:    ✓ O3, LTO, fast math enabled")
        message(STATUS "                 ✓ Maximum performance, production-ready")
        message(STATUS "                 ✓ No debug information")
        
    elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
        set(CXX_FLAGS
            ${RELWITHDEBINFO_FLAGS}
            ${CXX_WARNINGS}
            ${CXX_DIAGNOSTICS}
        )
        message(STATUS "RelWithDebInfo:  ✓ O2, debug symbols with optimization")
        message(STATUS "                 ✓ Ideal for profiling optimized code")
        
    else()
        message(FATAL_ERROR "Unsupported build type: ${CMAKE_BUILD_TYPE}")
    endif()
endif()

# ============================================================================
# Sanitizer Configuration
# ============================================================================
set(LINK_FLAGS "")
if(ENABLE_SANITIZERS)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        message(WARNING "⚠️  Sanitizers enabled with Release build - expect 2-3x slowdown")
    endif()
    list(APPEND CXX_FLAGS ${SANITIZER_FLAGS})
    list(APPEND LINK_FLAGS ${SANITIZER_FLAGS})
    message(STATUS "Sanitizers:      ✓ ENABLED (AddressSanitizer, UBSan, etc.)")
else()
    message(STATUS "Sanitizers:      ✗ Disabled")
endif()

message(STATUS "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
message(STATUS "")

# ============================================================================
# Output Directories
# ============================================================================
set(OUTPUT_DIR ${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR})
message(STATUS "📁 Output directory: ${OUTPUT_DIR}")

# ============================================================================
# Configuration File Generation from YAML
# Automatically generates C++ header from config.yaml
# ============================================================================
set(CONFIG_YAML "${CMAKE_SOURCE_DIR}/../config.yaml")
set(GENERATED_CONFIG_HPP "${CMAKE_BINARY_DIR}/generated_config.hpp")

if(NOT EXISTS ${CONFIG_YAML})
    message(FATAL_ERROR "❌ config.yaml not found at ${CONFIG_YAML}")
endif()

message(STATUS "📄 Parsing config.yaml...")

# Read YAML configuration file
file(READ "${CONFIG_YAML}" CONFIG_TEXT)

# ============================================================================
# YAML Extraction Patterns
# ============================================================================

include(../cmake/functions/GenerateConfig.cmake)
generate_config_header("${CONFIG_YAML}" "${GENERATED_CONFIG}")



# Display configuration summary
message(STATUS "=== Generated Configuration Summary ===")
message(STATUS "Config file: ${GENERATED_CONFIG_HPP}")
message(STATUS "")
message(STATUS "Compile-time DG Configuration:")
message(STATUS "  Order      = ${ORDER_VALUE}")
message(STATUS "  Dimension  = ${DIM_VALUE}")
message(STATUS "  DOFs       = ${DOFS_VALUE}")
message(STATUS "  Equation   = ${EQUATION_VALUE} (${EQUATION_ENUM})")
message(STATUS "  Scenario   = ${SCENARIO_VALUE}")
message(STATUS "")
message(STATUS "Simulation Parameters:")
message(STATUS "  End Time       = ${END_TIME_VALUE}")
message(STATUS "  Grid Elements  = ${GRID_ELEMENTS_VALUE}")
message(STATUS "  Grid Size      = ${GRID_SIZE_VALUE}")
message(STATUS "  Max Depth      = ${MAXDEPTH_VALUE}")
message(STATUS "")
message(STATUS "Solver Parameters:")
message(STATUS "  Courant Number    = ${COURANT_VALUE}")
message(STATUS "  Time Integrator   = ${TIMEINT_VALUE} (${TIMEINT_ENUM})")
message(STATUS "=======================================")

# ============================================================================
# Find Dependencies
# ============================================================================
find_package(TBB REQUIRED)
find_package(VTK REQUIRED COMPONENTS
    CommonCore
    CommonDataModel
    IOXML
    IOLegacy
    ImagingCore
)

message(STATUS "TBB found: ${TBB_VERSION}")
message(STATUS "VTK found: ${VTK_VERSION}")

# ============================================================================
# Build Examples
# ============================================================================
set(EXAMPLES_DIR "./")
file(GLOB EXAMPLE_FILES "${EXAMPLES_DIR}/*.e.cpp")

if(NOT EXAMPLE_FILES)
    message(WARNING "No example files (*.e.cpp) found in ${EXAMPLES_DIR}")
endif()

foreach(EXAMPLE_FILE ${EXAMPLE_FILES})
    get_filename_component(TARGET_NAME ${EXAMPLE_FILE} NAME_WE)
    
    # Create executable
    add_executable(${TARGET_NAME} ${EXAMPLE_FILE})
    
    # Include directories
    target_include_directories(${TARGET_NAME} 
        PUBLIC 
            ../include 
            ../lib 
            ${CMAKE_BINARY_DIR}
        PRIVATE 
            ${VTK_INCLUDE_DIRS}
    )
    
    # Compiler options
    target_compile_options(${TARGET_NAME} PRIVATE ${CXX_FLAGS})
    
    # Linker options
    target_link_options(${TARGET_NAME} PRIVATE ${LINK_FLAGS})
    
    # Libraries
    target_link_libraries(${TARGET_NAME} 
        PRIVATE 
            TBB::tbb 
            ${VTK_LIBRARIES}
    )
    
    # Ensure config header is generated first
    add_dependencies(${TARGET_NAME} generated_config_header)
    
    message(STATUS "Added target: ${TARGET_NAME}")
endforeach()

# ============================================================================
# Build Summary
# ============================================================================
message(STATUS "")
message(STATUS "=== Build Configuration Summary ===")
message(STATUS "Build Type:         ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard:       C++${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler:           ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "Sanitizers:         ${ENABLE_SANITIZERS}")
message(STATUS "Output Directory:   ${OUTPUT_DIR}")
message(STATUS "Parallel Jobs:      ${CMAKE_BUILD_PARALLEL_LEVEL}")
message(STATUS "====================================")
