# Definition of the minimum required cmake Version
cmake_minimum_required(VERSION 3.12)
# Definition of the Project
# Later you can access the project variable like ${CFDLAB_SOURCE_DIR}
project(CFDLAB VERSION 1.0)

# Define all configuration options
#option(option1 "compile using this option" ON)

# Definition of the C++ Standard 
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_BUILD_PARALLEL_LEVEL 8)

# Build type configurations
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
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

# TODO: Not working
# add_definitions("-DLSAN_SUPPRESSIONS_PATH=\"${CMAKE_SOURCE_DIR}/lsan.supp\"")

# Creating the executable of our project and the required dependencies
# the executable is called fluidchen

file(GLOB files src/*.cpp)
add_executable(fluidchen ${files})

# You can find package like
# find_package(MPI)
# Require a package
find_package(MPI REQUIRED)
# Find a package with different components e.g. BOOST
# find_package(Boost COMPONENTS filesystem REQUIRED)

# VTK Library
find_package(VTK REQUIRED COMPONENTS
  CommonCore
  CommonDataModel
  IOImage
  IOLegacy
  RenderingCore
  RenderingOpenGL2
)
message (STATUS "VTK_VERSION: ${VTK_VERSION}")
include(${VTK_USE_FILE})

# Add include directory
target_include_directories(fluidchen PUBLIC include)

# Add libmorton directory
target_include_directories(fluidchen PUBLIC libmorton)

# Add Compiler warnings (Try to write clean code!)
target_compile_options(fluidchen PRIVATE ${CXX_FLAGS})

target_link_options(fluidchen PRIVATE ${LINK_FLAGS})

# if you use external libraries you have to link them like
target_link_libraries(fluidchen PRIVATE MPI::MPI_CXX)
target_link_libraries(fluidchen PRIVATE ${VTK_LIBRARIES})

# If you write tests, you can include your subdirectory (in this case tests) as done here
# Testing

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CXX_FLAGS:\n ${CXX_FLAGS}")
message(STATUS "Sanitizers: ${ENABLE_SANITIZERS}")
