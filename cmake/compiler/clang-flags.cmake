# ----------------------------
# Compiler flags
# ----------------------------
set(AMR_WARNINGS
    -fvisibility=hidden
    -pedantic
    -Wall
    -Wconversion
    -Wdangling-else
    -Wdouble-promotion
    -Wpadded
    -Wredundant-decls
    -Wextra
    -Wfloat-equal
    -Wformat
    -Winvalid-pch
    -Wmisleading-indentation
    -Wnull-dereference
    -Wodr
    -Wpointer-arith
    -Wreturn-local-addr
    -Wshadow
    -Wswitch-default
    -Wswitch-enum
    -Wuninitialized
    -Wvla
)

set(AMR_DIAGNOSTICS
	-fdiagnostics-color=auto
	-fdiagnostics-show-template-tree
    # -ftime-report
)

set(AMR_DEBUG_INFO
	-fno-omit-frame-pointer
	-ggdb3
)

set(AMR_DEBUG_FLAGS
	-ffinite-math-only
	-fno-inline
	-march=native
	-O0
)
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64|ARM64")
    list(APPEND AMR_DEBUG_FLAGS -mavx)
endif()

set(AMR_RELEASE_FLAGS
	-fno-math-errno
	-ffast-math
	-fno-trapping-math
	-march=native
	-O3
)
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64|ARM64")
    list(APPEND AMR_RELEASE_FLAGS -mavx)
endif()

set(AMR_RELWITHDEBINFO_FLAGS
	-fno-math-errno
	-fno-trapping-math
	-fstrength-reduce
	-march=native
	-O2
)
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64|ARM64")
    list(APPEND AMR_RELWITHDEBINFO_FLAGS -mavx)
endif()

set(AMR_SANITIZERS
	-fsanitize=address
	-fsanitize=bounds
	-fsanitize=float-cast-overflow
	-fsanitize=float-divide-by-zero
	-fsanitize=integer-divide-by-zero
	-fsanitize=null
	-fsanitize=signed-integer-overflow
	-fsanitize=undefined
)

if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64|ARM64")
    list(APPEND AMR_SANITIZERS -fsanitize=leak)
endif()

set(CONSTEXPR_LIMIT_FLAGS)
