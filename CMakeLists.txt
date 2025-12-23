cmake_minimum_required(VERSION 3.25)
project(gpu_amr VERSION 0.1 LANGUAGES CXX)

# ----------------------------
# Global settings
# ----------------------------
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_BUILD_PARALLEL_LEVEL 12)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
endif()

# ----------------------------
# Options
# ----------------------------
option(ENABLE_SANITIZERS "Enable runtime sanitizers (Debug only)" ON)

# ----------------------------
# Logging
# ----------------------------
set(LOG_LEVEL "INFO" CACHE STRING "Logging level")

set_property(CACHE LOG_LEVEL PROPERTY STRINGS
    TRACE DEBUG INFO WARNING ERROR FATAL OFF)

# ----------------------------
# Output dir
# ----------------------------
set(OUTPUT_DIR_EXAMPLES   ${CMAKE_BINARY_DIR}/bin/examples/$<CONFIG>)
set(OUTPUT_DIR_BENCHMARKS ${CMAKE_BINARY_DIR}/bin/benchmarks)

# ----------------------------
# Dependencies (global)
# ----------------------------
find_package(TBB REQUIRED)

# ----------------------------
# Core code target (headers-only for now)
# ----------------------------
add_library(gpu_amr INTERFACE)

target_compile_features(gpu_amr INTERFACE cxx_std_23)

target_include_directories(gpu_amr INTERFACE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/lib
)

target_link_libraries(gpu_amr INTERFACE
    TBB::tbb
)

# ----------------------------
# Compiler flags
# ----------------------------
set(PROJECT_WARNINGS
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

set(PROJECT_DIAGNOSTICS
	-fconcepts-diagnostics-depth=3
	-fdiagnostics-color=auto
	-fdiagnostics-path-format=inline-events
	-fdiagnostics-show-caret
	-fdiagnostics-show-template-tree
)

set(PROJECT_DEBUG_INFO
	-fno-omit-frame-pointer
	-fvar-tracking
	-fvar-tracking-assignments
	-ggdb3
	-gvariable-location-views
	-ginline-points
	-gstatement-frontiers
)

set(PROJECT_DEBUG_FLAGS
	-ffinite-math-only
	-fmax-errors=15
	-fno-eliminate-unused-debug-symbols
	-fno-inline
	-fno-default-inline
	-march=native
	-mavx
	-O0
)

set(PROJECT_RELEASE_FLAGS
	-fno-math-errno
	-fno-trapping-math
	-fstrength-reduce
	-march=native
	-mavx
	-O3
)

set(REL_DEB_INFO_CXX_FLAGS
	-fno-math-errno
	-fno-trapping-math
	-fstrength-reduce
	-march=native
	-mavx
	-O2
)

set(PROJECT_SANITIZERS
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

set(CONSTEXPR_LIMIT_FLAGS
    -fconstexpr-ops-limit=10000000000
    -fconstexpr-loop-limit=1048576
)

# ----------------------------
# Policy targets
# ----------------------------
add_library(amr_examples INTERFACE)
add_library(amr_benchmarks INTERFACE)
add_library(amr_tests INTERFACE)

target_compile_definitions(amr_examples INTERFACE
    AMR_LOG_LEVEL=${LOG_LEVEL}
)

target_compile_definitions(amr_benchmarks INTERFACE
    AMR_LOG_LEVEL=OFF
)

target_compile_definitions(amr_tests INTERFACE
    AMR_LOG_LEVEL=WARNING
)

foreach(tgt amr_examples amr_benchmarks amr_tests)
    target_compile_options(${tgt} INTERFACE
        ${PROJECT_WARNINGS}
        ${PROJECT_DIAGNOSTICS}
        ${CONSTEXPR_LIMIT_FLAGS}

        $<$<CONFIG:Debug>:${PROJECT_DEBUG_FLAGS} ${PROJECT_DEBUG_INFO}>
        $<$<CONFIG:Release>:${PROJECT_RELEASE_FLAGS}>
        $<$<CONFIG:RelWithDebInfo>:${REL_DEB_INFO_CXX_FLAGS} ${PROJECT_DEBUG_INFO}>

        $<$<AND:$<CONFIG:Debug>,$<BOOL:${ENABLE_SANITIZERS}>>:${PROJECT_SANITIZERS}>
    )

    target_link_options(${tgt} INTERFACE
        $<$<AND:$<CONFIG:Debug>,$<BOOL:${ENABLE_SANITIZERS}>>:${PROJECT_SANITIZERS}>
    )
endforeach()

# ----------------------------
# CMake Subdirectories
# ----------------------------

add_subdirectory(examples)
add_subdirectory(tests)
add_subdirectory(benchmark)

# ----------------------------
# Status
# ----------------------------
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Sanitizers enabled (Debug only): ${ENABLE_SANITIZERS}")
message(STATUS "Examles log level: ${LOG_LEVEL}")
