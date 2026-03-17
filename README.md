# GPU-Driven Adaptative Mesh Refinement

## Introduction

Adaptative Mesh Refinement (AMR) framework.

## Build

The examples for this project in the `examples/` directory. Files with a `.e.cpp` extension, are automatically detected by CMake.\
The benchmarks for this project in the `benchmark/` directory. Files with a `.b.cpp` extension, are automatically detected by CMake.\
The testes for this project in the `test/` directory. Files with a `.t.cpp` extension, are automatically detected by CMake.

The project is built with CMake. In order to allow multiple compiler toolchains
an appropriate driver is required.
CMakePresets.json contains defaults for the most common configurations and new
configurations should be added for other compilers, toolchains or common
configurations.

The project can be build using theses presets, for example, from the root directory
1. Generate the configuration and build output directory: `cmake --preset $(preset) [-DOPTIONS=OPT...]`
2. Build the project: `cmake --build --preset $(preset)`\
If compilation is successful, the build output will be located in
`build/$(preset)/bin/examples/$(example).o`.\
Available presets can be listed with `cmake --list-presets`.

### Build Options

The optional build flags `OPTIONS` are used to enable parts of the project that might require dependencies or might not make sense always.
The supported arguments are
- `ENABLE_SANITIZERS={OFF,ON}`: Disables/Enables the use of sanitizers. Defaults to `ON`.
- `LOG_LEVEL={TRACE,DEBUG,INFO,WARNING,ERROR,FATAL,OFF}`: Control the logging level. Defaults to `INFO`.
- `EXECUTION={PAR,SEQ}`: Control the logging level. Defaults to `SEQ`.

### Compile Time Configuration

Compile time configuration of the solver is done through `config/config.yaml`
and a parser in `cmake/functions/GenerateConfig.cmake`. The generated
configuration file is located in `include/config/generated_config.hpp`.

## Testing

The tests for this project in the `tests/` directory. Tests with a `.t.cpp` extension, are automatically detected by CMake.\
GoogleTest has been used as the testing framework.

Test can be run after building. For example, from the project root directory,
after a successful build:
1. `ctest --test-dir build/$(preset)/tests`

## Performance Analysis

Due to the premature status of the project, performance will not be analyzed in
depth.
Simple performance metrics will be collected to identify potential issues and regression analysis.

More information and benchmark logs can be found in `./reports`

## License
This project is licensed under the [MIT License](./LICENSE).

For more information, please see the [LICENSE](./LICENSE) file.
