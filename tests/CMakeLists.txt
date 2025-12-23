enable_testing()

include(FetchContent)

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
)

FetchContent_MakeAvailable(googletest)
include(GoogleTest)

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS *.t.cpp)

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS *.t.cpp)
foreach(src ${TEST_SOURCES})
    get_filename_component(target ${src} NAME_WE)
    add_executable(${target} ${src})
    target_link_libraries(${target} PRIVATE gpu_amr amr_tests GTest::gtest_main)
    gtest_discover_tests(${target})
endforeach()

add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
