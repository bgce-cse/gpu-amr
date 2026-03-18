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
	-mavx
	-O0
)

set(AMR_RELEASE_FLAGS
	-fno-math-errno
	-ffast-math
	-fno-trapping-math
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

set(CONSTEXPR_LIMIT_FLAGS)
