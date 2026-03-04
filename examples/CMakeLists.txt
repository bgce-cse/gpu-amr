file(GLOB EXAMPLE_SOURCES CONFIGURE_DEPENDS
    *.e.cpp
)

foreach(src ${EXAMPLE_SOURCES})
    get_filename_component(target ${src} NAME_WE)
    add_executable(${target} ${src})
    target_link_libraries(${target} PRIVATE
    gpu_amr
    gpu_solver
    amr_examples
    )

    if(target MATCHES "fvm_solver_integration_gpu")
        target_link_libraries(${target} PRIVATE gpu_solver)
    endif()

    set_target_properties(${target} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR_EXAMPLES}
    )
endforeach()
