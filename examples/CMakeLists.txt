cmake_minimum_required(VERSION 3.30)

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
	-mavx
	-O0
)

set(RELEASE_CXX_FLAGS
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

set(EXAMPLES_DIR "./")
find_package(TBB REQUIRED)

# Generate executable for all .e.cpp
file(GLOB EXAMPLE_FILES "${EXAMPLES_DIR}/*.e.cpp")
foreach(EXAMPLE_FILE ${EXAMPLE_FILES})
    get_filename_component(TARGET_NAME ${EXAMPLE_FILE} NAME_WE)
    add_executable(${TARGET_NAME} ${EXAMPLE_FILE})
    target_include_directories(${TARGET_NAME} PUBLIC ../include)
    target_compile_options(${TARGET_NAME} PRIVATE ${CXX_FLAGS})
    target_link_options(${TARGET_NAME} PRIVATE ${LINK_FLAGS})
    target_link_libraries(${TARGET_NAME} PRIVATE TBB::tbb)
endforeach()

# Build messages
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Output directory: ${OUTPUT_DIR}")
message(STATUS "Sanitizers: ${ENABLE_SANITIZERS}")
