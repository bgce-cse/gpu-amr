cmake_minimum_required(VERSION 3.25)
project(GPU_AMR VERSION 0.1 LANGUAGES CXX)

# ----------------------------
# Basic project settings
# ----------------------------
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_BUILD_PARALLEL_LEVEL 8)

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
set(LOG_LEVEL "INFO" CACHE STRING
    "Logging level: TRACE, DEBUG, INFO, WARNING, ERROR, FATAL, OFF")

set_property(CACHE LOG_LEVEL PROPERTY STRINGS
    TRACE DEBUG INFO WARNING ERROR FATAL OFF)

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

# ----------------------------
# Central INTERFACE target
# ----------------------------
add_library(project_examples INTERFACE)

set(CONSTEXPR_LIMIT_FLAGS
    -fconstexpr-ops-limit=10000000000
    -fconstexpr-loop-limit=1048576
)



target_compile_options(project_examples INTERFACE
    -fconstexpr-ops-limit=100000000
    ${PROJECT_WARNINGS}
    ${PROJECT_DIAGNOSTICS}
    ${CONSTEXPR_LIMIT_FLAGS}

    # Debug
    $<$<CONFIG:Debug>:${PROJECT_DEBUG_FLAGS} ${PROJECT_DEBUG_INFO}>

    # Release
    $<$<CONFIG:Release>:${PROJECT_RELEASE_FLAGS}>

    # RelWithDebInfo
    $<$<CONFIG:RelWithDebInfo>:${REL_DEB_INFO_CXX_FLAGS} ${PROJECT_DEBUG_INFO}>

    # Sanitizers: Debug only
    $<$<AND:$<CONFIG:Debug>,$<BOOL:${ENABLE_SANITIZERS}>>:${PROJECT_SANITIZERS}>
)

target_compile_definitions(project_examples INTERFACE
    AMR_LOG_LEVEL=${LOG_LEVEL}
)

target_link_options(project_examples INTERFACE
    $<$<AND:$<CONFIG:Debug>,$<BOOL:${ENABLE_SANITIZERS}>>:${PROJECT_SANITIZERS}>
)

target_include_directories(project_examples INTERFACE
    ${PROJECT_SOURCE_DIR}/../include
    ${PROJECT_SOURCE_DIR}/../lib
)

add_library(project_benchmarks INTERFACE)

target_compile_options(project_benchmarks INTERFACE
    ${PROJECT_WARNINGS}
    ${PROJECT_DIAGNOSTICS}
    ${CONSTEXPR_LIMIT_FLAGS}
    ${PROJECT_RELEASE_FLAGS}
)

target_compile_definitions(project_benchmarks INTERFACE
    AMR_LOG_LEVEL=OFF
)

target_link_options(project_benchmarks INTERFACE
)

target_include_directories(project_benchmarks INTERFACE
    ${PROJECT_SOURCE_DIR}/../include
    ${PROJECT_SOURCE_DIR}/../lib
)

# ----------------------------
# Output directory
# ----------------------------
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/$<CONFIG>)
message(STATUS "Output directory pattern: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")

# ----------------------------
# Dependencies
# ----------------------------
find_package(TBB REQUIRED)

# ----------------------------
# Example executables
# ----------------------------
file(GLOB EXAMPLE_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.e.cpp
)

foreach(src ${EXAMPLE_SOURCES})
    get_filename_component(name ${src} NAME_WE)
    add_executable(${name} ${src})

    target_link_libraries(${name} PRIVATE project_examples TBB::tbb)
endforeach()

# ----------------------------
# Benchmark executables
# ----------------------------
file(GLOB BENCH_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.b.cpp
)

foreach(src ${BENCH_SOURCES})
    get_filename_component(name ${src} NAME_WE)
    add_executable(${name} ${src})

    target_link_libraries(${name} PRIVATE project_benchmarks TBB::tbb)
endforeach()

# ----------------------------
# Status
# ----------------------------
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Sanitizers enabled (Debug only): ${ENABLE_SANITIZERS}")
message(STATUS "Log level: ${AMR_LOG_LEVEL}")
