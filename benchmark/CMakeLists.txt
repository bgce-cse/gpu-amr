file(GLOB BENCHMARK_SOURCES CONFIGURE_DEPENDS
    *.b.cpp
)

foreach(src ${BENCHMARK_SOURCES})
    get_filename_component(target ${src} NAME_WE)

    add_executable(${target} ${src})

    target_link_libraries(${target} PRIVATE
        gpu_amr
        amr_benchmarks
    )

    set_target_properties(${target} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR_BENCHMARKS}
    )
endforeach()
