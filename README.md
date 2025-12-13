# GPU-Driven Adaptative Mesh Refinement

## Introduction

Adaptative Mesh Refinement (AMR) framework.

## Build

The examples for this project in the `examples/` directory, which have a `.e.cpp` extension, are automatically detected by CMake.

These examples can be compiled with cmake using the provided CMakeLists.txt.\
One possible way to compile is:
1. Create a build directory and move into it: `mkdir build && cd build`
2. Run cmake: `cmake -DCMAKE_BUILD_TYPE={Debug,Release,RelWithDebInfo} [-DOPTIONS=ON...] ..`
3. Run the generated Makefile: `make`
If compilation is successful, the build output will be located in
`bin/{Debug,Release,RelWithDebInfo}/{example}`.
Refer to [Compile time configuration](#Compile_time_configuration) for further details.

### Build Options

The optional build flags `OPTIONS` are used to enable parts of the project that might require dependencies or might not make sense always.
The supported arguments are
- `ENABLE_SANITIZERS={OFF,ON}`:Disables/Enables the use of sanitizers. Defaults to `ON`.

### Compile Time Configuration

Work in progress

## Testing

No testing yet.

## License
This project is licensed under the [MIT License](./LICENSE).

For more information, please see the [LICENSE](./LICENSE) file.
