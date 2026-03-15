function(add_group suffix interface_target)
    file(GLOB sources CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.${suffix}.cpp
    )

    foreach(src ${sources})
        get_filename_component(name ${src} NAME_WE)
        add_executable(${name} ${src})

        target_link_libraries(${name} PRIVATE
            gpu_amr
            ${interface_target}
        )
    endforeach()
endfunction()
