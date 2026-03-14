# ----------------------------
# Compiler flags
# ----------------------------
set(AMR_WARNINGS
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
    -Wpadded
    -Wredundant-decls
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

set(AMR_DIAGNOSTICS
	-fconcepts-diagnostics-depth=3
	-fdiagnostics-color=auto
	-fdiagnostics-path-format=inline-events
	-fdiagnostics-show-caret
	-fdiagnostics-show-template-tree
    # -ftime-report
)

set(AMR_DEBUG_INFO
	-fno-omit-frame-pointer
	-fvar-tracking
	-fvar-tracking-assignments
	-ggdb3
	-gvariable-location-views
	-ginline-points
	-gstatement-frontiers
)

set(AMR_DEBUG_FLAGS
	-ffinite-math-only
	-fmax-errors=15
	-fno-eliminate-unused-debug-symbols
	-fno-inline
	-fno-default-inline
	-march=native
	-mavx
	-O0
)

set(AMR_RELEASE_FLAGS
	-fno-math-errno
	-ffast-math
	-fno-trapping-math
	-fstrength-reduce
	-march=native
	-mavx
	-O3
)

set(AMR_RELWITHDEBINFO_FLAGS
	-fno-math-errno
	-fno-trapping-math
	-fstrength-reduce
	-march=native
	-mavx
	-O2
)

set(AMR_SANITIZERS
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
