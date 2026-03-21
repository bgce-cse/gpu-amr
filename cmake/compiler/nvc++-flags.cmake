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
    -Werror
    -Wredundant-decls
    -Wextra
    -Wfloat-equal
    -Wformat
    -Wpointer-arith
    -Wshadow
    -Wswitch-enum
    -Wuninitialized
    -Wvla
)

set(AMR_DIAGNOSTICS
	-fdiagnostics-color=auto
)

set(AMR_DEBUG_INFO
	-fno-omit-frame-pointer
)

set(AMR_DEBUG_FLAGS
	-ffinite-math-only
	-fmax-errors=15
	-march=native
	-mavx
	-O0
)

set(AMR_RELEASE_FLAGS
    -stdpar=gpu
	-march=native
	-mavx
	-O3
)

set(AMR_RELWITHDEBINFO_FLAGS
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
