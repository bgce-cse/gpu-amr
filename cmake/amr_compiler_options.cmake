# ----------------------------
# Toolchain selection
# ----------------------------
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    include(${CMAKE_CURRENT_LIST_DIR}/compiler/gcc-flags.cmake)

elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    include(${CMAKE_CURRENT_LIST_DIR}/compiler/clang-flags.cmake)

elseif(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    include(${CMAKE_CURRENT_LIST_DIR}/compiler/clang-flags.cmake)

elseif(CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
    include(${CMAKE_CURRENT_LIST_DIR}/compiler/nvc++-flags.cmake)

else()
    message(FATAL_ERROR "Unsupported compiler")
endif()

# ----------------------------
# Global compiler options target
# ----------------------------
add_library(amr_compiler_options INTERFACE)

target_compile_options(amr_compiler_options INTERFACE
    ${AMR_WARNINGS}
    ${AMR_DIAGNOSTICS}
    ${CONSTEXPR_LIMIT_FLAGS}

    $<$<CONFIG:Debug>:${AMR_DEBUG_FLAGS} ${AMR_DEBUG_INFO}>
    $<$<CONFIG:Release>:${AMR_RELEASE_FLAGS}>
    $<$<CONFIG:RelWithDebInfo>:${AMR_RELWITHDEBINFO_FLAGS} ${AMR_DEBUG_INFO}>

    $<$<BOOL:${ENABLE_SANITIZERS}>:${AMR_SANITIZERS}>
)

target_link_options(amr_compiler_options INTERFACE
    $<$<BOOL:${ENABLE_SANITIZERS}>:${AMR_SANITIZERS}>
)
