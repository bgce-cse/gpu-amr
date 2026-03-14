# Compile-Time API Documentation: Basis, Globals, and Equations

This document provides a comprehensive guide to accessing compile-time available data from the DG (Discontinuous Galerkin) helper modules: basis, globals, and equations.

**Key Principle:** All components documented here are **compile-time constants** (`constexpr`), meaning they can be accessed, computed, and verified at **compile-time** using `static_assert` and `constexpr` expressions.

---

## Table of Contents

1. [Configuration Parameters](#configuration-parameters)
2. [Basis Functions & Quadrature](#basis-functions--quadrature)
3. [Global Configuration](#global-configuration)
4. [Equation Components](#equation-components)
5. [Container Operations](#container-operations)
6. [Complete Examples](#complete-examples)

---

## Configuration Parameters

Configuration values are generated at build-time from `config.yaml` and available as `constexpr` constants in the `amr::config` namespace (via `generated_config.hpp`).

### Available Constants

```cpp
#include "generated_config.hpp"

namespace amr::config {
    // Polynomial and spatial parameters
    constexpr unsigned int Order;           // Polynomial order for DG basis (e.g., 1, 2, 3)
    constexpr unsigned int Dim;             // Spatial dimension (2D or 3D)
    constexpr unsigned int DOFs;            // Degrees of freedom per cell

    // Equation configuration
    constexpr std::string_view Equation;    // Equation type: "advection", "euler", etc.
    constexpr std::string_view Scenario;    // Scenario name: "gaussian_wave", etc.

    // Simulation parameters
    constexpr double EndTime;               // Total simulation time
    constexpr unsigned int GridElements;    // Grid elements per dimension
    constexpr double GridSize;              // Physical domain size
    constexpr double CourantNumber;         // CFL number for stability
    constexpr std::string_view TimeIntegrator; // Time integration scheme
}
```

### Usage Example

```cpp
#include "generated_config.hpp"

// Verify configuration at compile-time
static_assert(amr::config::Order == 1, "Expected first-order polynomial");
static_assert(amr::config::Dim == 2, "Expected 2D simulation");
static_assert(amr::config::DOFs == 3, "Expected 3 DOFs for Order=1, Dim=2");

// Use in constexpr contexts
constexpr unsigned int my_order = amr::config::Order;
constexpr unsigned int my_dim = amr::config::Dim;
```

---

## Basis Functions & Quadrature

The basis module provides compile-time Lagrange polynomial basis functions with Gauss-Legendre quadrature.

### Creating a Basis

```cpp
#include "dg_helpers/basis/basis.hpp"

// Create a basis with specific Order and Dimension
using Basis_Order1_2D = amr::basis::Basis<1, 2>;  // Order=1, Dim=2, DOFs=3
using Basis_Order2_2D = amr::basis::Basis<2, 2>;  // Order=2, Dim=2, DOFs=6
using Basis_Order3_2D = amr::basis::Basis<3, 2>;  // Order=3, Dim=2, DOFs=10

using Basis_Order1_3D = amr::basis::Basis<1, 3>;  // Order=1, Dim=3, DOFs=4
using Basis_Order2_3D = amr::basis::Basis<2, 3>;  // Order=2, Dim=3, DOFs=10

// Or use configuration parameters (dynamically set from config.yaml)
using ConfiguredBasis = amr::basis::Basis<amr::config::Order, amr::config::Dim>;
```

**Note:** DOFs (Degrees of Freedom) depend on Order and Dimension:
- Order=1, Dim=2: 3 DOFs
- Order=2, Dim=2: 6 DOFs  
- Order=3, Dim=2: 10 DOFs
- Order=1, Dim=3: 4 DOFs
- Order=2, Dim=3: 10 DOFs
- Order=3, Dim=3: 20 DOFs

### Basis Static Members

All members are `static constexpr` and compile-time accessible:

```cpp
// Basic information
static constexpr unsigned int order = Order;           // Polynomial order
static constexpr unsigned int dimensions = Dim;       // Spatial dimension

// Quadrature data (references to GaussLegendre points/weights)
static constexpr const auto& quadpoints  = ...;       // Vector<Order> of quadrature points
static constexpr const auto& quadweights = ...;       // Vector<Order> of quadrature weights

// Access number of quadrature points
constexpr unsigned int num_quad_points = MyBasis::quadpoints.elements();
static_assert(num_quad_points == MyBasis::order, "Must have Order quadrature points");
```

### Quadrature Points and Weights

```cpp
#include "dg_helpers/basis/basis.hpp"

using Basis = amr::basis::Basis<1, 2>;

// Accessing individual quadrature points and weights
constexpr auto test_quadrature = []() {
    double sum_weights = 0.0;
    for (unsigned int i = 0; i < Basis::quadpoints.elements(); ++i) {
        double pt = Basis::quadpoints[i];      // Quadrature point i
        double wt = Basis::quadweights[i];     // Quadrature weight i
        sum_weights += wt;
    }
    return sum_weights;
}();

// For Gauss-Legendre on [0,1], weights sum to 1.0
static_assert(test_quadrature > 0.99 && test_quadrature < 1.01, 
              "Weights should sum to ~1.0");
```

### Creating Face Kernels

Face kernels are 1D basis function evaluations at element boundaries:

```cpp
using Basis = amr::basis::Basis<amr::config::Order, amr::config::Dim>;

// Create face kernels at x=0.0 and x=1.0 (element boundaries)
constexpr auto kernel_at_0 = Basis::create_face_kernel(0.0);
constexpr auto kernel_at_1 = Basis::create_face_kernel(1.0);

// Verify they have the correct size
static_assert(
    kernel_at_0.elements() == amr::config::Order,
    "Face kernel should have Order entries"
);

// Access individual basis function values at a boundary
constexpr double phi_0_at_left = kernel_at_0[0];   // First basis function at x=0
constexpr double phi_1_at_left = kernel_at_0[1];   // Second basis function at x=0
```

### Basis Evaluation

Evaluate the basis at a point using stored coefficients:

```cpp
using Basis = amr::basis::Basis<1, 2>;
using vector_t = amr::containers::static_vector<double, 2>;
using tensor_t = /* coefficient tensor type */;

// Create a coefficient tensor (e.g., for storing DG solution)
tensor_t coeffs{};
// ... populate coefficients ...

// Evaluate basis at a point
vector_t point{0.5, 0.5};
constexpr double result = Basis::evaluate_basis(coeffs, point);
```

### Lagrange Polynomial Evaluation (Barycentric Method)

The Lagrange basis can be evaluated using the barycentric form, which is **O(N) instead of O(N²)** and more numerically stable.

```cpp
#include "dg_helpers/basis/lagrange.hpp"

using Lagrange = amr::basis::Lagrange<3>;  // 3 nodes

// Get quadrature points
using Basis = amr::basis::Basis<1, 2>;
const auto& points = Basis::quadpoints;

// Compute barycentric weights at compile-time (done once)
constexpr auto weights = Lagrange::compute_barycentric_weights(points);

// Evaluate barycentric Lagrange basis
constexpr double x = 0.5;
constexpr auto phi_0 = Lagrange::evaluate_barycentric(points, weights, 0, x);
constexpr auto phi_1 = Lagrange::evaluate_barycentric(points, weights, 1, x);

// Compute derivatives using barycentric form (also O(N))
constexpr auto dphi_0 = Lagrange::derivative_barycentric(points, weights, 0, x);
constexpr auto dphi_1 = Lagrange::derivative_barycentric(points, weights, 1, x);

// Verify partition of unity: all basis functions sum to 1.0
constexpr auto partition_test = []() {
    double sum = 0.0;
    for (unsigned int i = 0; i < 3; ++i) {
        sum += Lagrange::evaluate_barycentric(points, weights, i, x);
    }
    return sum;
}();
static_assert(partition_test > 0.99 && partition_test < 1.01,
              "Lagrange basis must partition unity");

// Verify derivative property: derivatives sum to 0
constexpr auto derivative_test = []() {
    double sum = 0.0;
    for (unsigned int i = 0; i < 3; ++i) {
        sum += Lagrange::derivative_barycentric(points, weights, i, x);
    }
    return sum;
}();
static_assert(derivative_test > -1e-10 && derivative_test < 1e-10,
              "Lagrange basis derivatives must sum to 0");
```

**Barycentric Method Advantages:**
- **Time Complexity:** O(N) vs O(N²) for standard Lagrange
- **Numerical Stability:** Better conditioned for high-order polynomials
- **Compile-Time Available:** All three methods are fully constexpr
  - `compute_barycentric_weights(points)` - Compute weights once
  - `evaluate_barycentric(points, weights, i, x)` - Fast evaluation
  - `derivative_barycentric(points, weights, i, x)` - Fast derivative

---

## Global Configuration

The `GlobalConfig` template provides compile-time access to basis, quadrature, kernels, and mass tensors, organized by polynomial order and spatial dimension.

### Creating GlobalConfig

```cpp
#include "dg_helpers/globals/global_config.hpp"

// Template: GlobalConfig<Order, Dim, PatchSize, HaloWidth>
using MyGlobalConfig = amr::global::GlobalConfig<
    1,        // Order (polynomial)
    2,        // Dim (spatial dimension)
    4,        // PatchSize (cells per patch dimension)
    1         // HaloWidth (halo region width)
>;

// Or use configuration parameters
using ConfiguredGlobalConfig = amr::global::GlobalConfig<
    amr::config::Order,
    amr::config::Dim,
    4,  // typical patch size
    1   // typical halo width
>;
```

### GlobalConfig Static Members

All members are `static constexpr`:

```cpp
using GC = amr::global::GlobalConfig<1, 2, 4, 1>;

// Quadrature data
static constexpr const auto& quad_points  = GC::quad_points;   // Points
static constexpr const auto& quad_weights = GC::quad_weights;  // Weights

// Basis
using Basis_t = GC::Basis;                 // Basis<Order, Dim>

// Face kernels (std::array<static_vector<double, Order>, 2>)
static constexpr const auto& face_kernels = GC::face_kernels;  // [0] at x=0, [1] at x=1

// Mass matrices (fully compile-time tensors)
static constexpr const auto& volume_mass      = GC::volume_mass;      // Main mass matrix
static constexpr const auto& inv_volume_mass  = GC::inv_volume_mass;  // Inverse
static constexpr const auto& surface_mass     = GC::surface_mass;     // 1D surface mass
static constexpr const auto& inv_surface_mass = GC::inv_surface_mass; // Inverse
```

### Accessing Face Kernels

```cpp
using GC = amr::global::GlobalConfig<1, 2, 4, 1>;

// Face kernels at two boundaries: x=0.0 and x=1.0
constexpr const auto& kernel_left  = GC::face_kernels[0];  // At x=0
constexpr const auto& kernel_right = GC::face_kernels[1];  // At x=1

// Each kernel is a vector of Order basis function evaluations
for (unsigned int i = 0; i < kernel_left.elements(); ++i) {
    constexpr double phi_i_left = kernel_left[i];   // Basis i at x=0
}
```

### Accessing Mass Tensors

Mass tensors have rank equal to `Dim`:

```cpp
using GC = amr::global::GlobalConfig<1, 2, 4, 1>;

// Volume mass tensor rank and size
constexpr auto rank = GC::volume_mass.rank();          // 2 for Dim=2
constexpr auto size = GC::volume_mass.elements();      // Total number of entries
constexpr auto dim0 = GC::volume_mass.size(0);         // Size along dimension 0
constexpr auto dim1 = GC::volume_mass.size(1);         // Size along dimension 1

// Access mass matrix entries
constexpr double M_00 = GC::volume_mass[0, 0];         // Entry at (0,0)
constexpr double M_01 = GC::volume_mass[0, 1];         // Entry at (0,1)

// Inverse mass tensor
constexpr double invM_00 = GC::inv_volume_mass[0, 0];  // Inverse entry
```

### Coordinate Transformations

GlobalConfig provides constexpr coordinate transformation functions:

```cpp
using GC = amr::global::GlobalConfig<1, 2, 4, 1>;
using vector_t = amr::containers::static_vector<double, 2>;

// Reference-to-global coordinate transformation
vector_t global_pt = GC::ref_to_global(center, ref_point, cell_size);

// Global-to-reference coordinate transformation
vector_t ref_pt = GC::global_to_ref(center, global_point, cell_size);

// Cell volume calculation
double volume = GC::cell_volume(cell_size);

// Cell area (surface area for 2D)
double area = GC::cell_area(cell_size);
```

### Linear Index to Local Coordinates

Convert patch-local linear indices to coordinate arrays:

```cpp
using GC = amr::global::GlobalConfig<1, 2, 4, 1>;

// Convert linear index to local coordinates (with halo)
constexpr auto coords_with_halo = GC::lin_to_local(linear_idx);

// Remove halo from coordinates
constexpr auto coords_no_halo = GC::rm_halo(coords_with_halo);
```

---

## Equation Components

Equations are implemented using the CRTP (Curiously Recurring Template Pattern) and provide compile-time flux calculations and eigenvalue computations.

### Advection Equation

```cpp
#include "dg_helpers/equations/advection.hpp"

// Template: Advection<NumDOFs, Order, Dim, Velocity, Scalar>
using MyAdvection = amr::equations::Advection<
    3,        // NumDOFs (degrees of freedom per cell)
    1,        // Order (polynomial order)
    2,        // Dim (spatial dimension)
    1.0,      // Velocity (advection speed)
    double    // Scalar type
>;

// Or use configuration parameters
using ConfiguredAdvection = amr::equations::Advection<
    amr::config::DOFs,
    amr::config::Order,
    amr::config::Dim,
    1.0,      // velocity value
    double
>;
```

### Advection Static Members

```cpp
using Advection = amr::equations::Advection<3, 1, 2, 1.0, double>;

// Verify the equation is a valid class
static_assert(
    std::is_class_v<Advection>,
    "Advection must instantiate as a valid class"
);

// Static constants
constexpr double velocity = Advection::velocity;  // 1.0

// Type information
using dof_t = typename Advection::dof_t;          // DOF tensor type
using flux_t = typename Advection::flux_t;        // Flux tensor type
```

### Equation Methods (Compile-Time Capable)

Advection (and other equations) provide methods that can be called in `constexpr` contexts:

```cpp
using Advection = amr::equations::Advection<3, 1, 2, 1.0, double>;
using dof_t = typename Advection::dof_t;

// Evaluate flux (constexpr-capable)
constexpr auto evaluate_flux = [](const dof_t& u) {
    return Advection::evaluate_flux(u);
};

// Compute maximum eigenvalue (constexpr-capable)
constexpr auto max_eigenvalue = [](const dof_t& u, unsigned int direction) {
    return Advection::max_eigenvalue(u, direction);
};

// Get initial values at a point (constexpr-capable)
constexpr auto initial_values = [](const vector_t& pos, double t) {
    return Advection::get_initial_values(pos, t);
};
```

---

## Container Operations

The library provides constexpr container types with operations:

### Static Vector

```cpp
#include "containers/static_vector.hpp"

// Create a vector at compile-time
constexpr amr::containers::static_vector<double, 3> v{};

// Access size
constexpr unsigned int size = v.elements();        // 3

// Access elements
constexpr double val = v[0];                       // First element
v[1] = 2.5;                                        // Assignment in constexpr lambda

// Container operations
constexpr auto v2 = v + v;                         // Addition
constexpr auto v3 = 2.0 * v;                       // Scalar multiplication
constexpr auto v4 = v / 2.0;                       // Division
```

### Static Tensor

```cpp
#include "containers/static_tensor.hpp"

// Create a 2D tensor
using Layout = amr::containers::static_layout<
    amr::containers::static_shape<3, 3>
>;
using Tensor = amr::containers::static_tensor<double, Layout>;

// Access rank and size
constexpr unsigned int rank = Tensor::rank();       // 2
constexpr unsigned int elems = Tensor::elements();  // 9
constexpr unsigned int sz0 = Tensor::size(0);      // 3
constexpr unsigned int sz1 = Tensor::size(1);      // 3

// Access elements
constexpr double m00 = T[0, 0];                     // Multi-index access
constexpr double m_linear = T[0];                   // Linear access
```

---

## Complete Examples

### Example 1: Verify Gauss-Legendre Quadrature

```cpp
#include "dg_helpers/basis/basis.hpp"
#include "generated_config.hpp"

namespace {
    // Create basis with configuration parameters
    using Basis = amr::basis::Basis<amr::config::Order, amr::config::Dim>;

    // Verify quadrature properties
    static_assert(
        Basis::quadpoints.elements() == amr::config::Order,
        "Must have Order quadrature points"
    );

    // Compute weight sum
    constexpr auto weight_sum = []() {
        double sum = 0.0;
        for (unsigned int i = 0; i < Basis::quadweights.elements(); ++i) {
            sum += Basis::quadweights[i];
        }
        return sum;
    }();

    // Verify weights sum to ~1.0 on [0,1]
    static_assert(
        weight_sum > 0.99 && weight_sum < 1.01,
        "Gauss-Legendre weights on [0,1] should sum to 1.0"
    );
}
```

### Example 2: Access GlobalConfig Mass Matrices

```cpp
#include "dg_helpers/globals/global_config.hpp"

namespace {
    using GC = amr::global::GlobalConfig<1, 2, 4, 1>;

    // Verify mass tensor properties
    static_assert(
        GC::volume_mass.rank() == 2,
        "Mass matrix for 2D should have rank 2"
    );

    static_assert(
        GC::volume_mass.elements() == 9,
        "Mass matrix for Order=1, Dim=2 should have 9 entries (3x3)"
    );

    // Create mass matrix and use in computation
    constexpr auto mass_matrix = []() {
        // Could implement matrix operations here
        return GC::volume_mass[0, 0];  // Just return one entry for this example
    }();
}
```

### Example 3: Verify Advection Equation

```cpp
#include "dg_helpers/equations/advection.hpp"
#include "generated_config.hpp"

namespace {
    using Adv = amr::equations::Advection<
        amr::config::DOFs,
        amr::config::Order,
        amr::config::Dim,
        1.0,
        double
    >;

    // Verify equation instantiation
    static_assert(
        std::is_class_v<Adv>,
        "Advection equation must be instantiable"
    );

    // Verify velocity parameter
    static_assert(
        Adv::velocity >= 0.99 && Adv::velocity <= 1.01,
        "Velocity should be approximately 1.0"
    );
}
```

### Example 4: Complete Integration Test

```cpp
#include "dg_helpers/basis/basis.hpp"
#include "dg_helpers/globals/global_config.hpp"
#include "dg_helpers/equations/advection.hpp"
#include "generated_config.hpp"

namespace test {
    // Step 1: Define all types
    using Basis = amr::basis::Basis<amr::config::Order, amr::config::Dim>;
    using GC = amr::global::GlobalConfig<amr::config::Order, amr::config::Dim, 4, 1>;
    using Adv = amr::equations::Advection<
        amr::config::DOFs, amr::config::Order, amr::config::Dim, 1.0, double
    >;

    // Step 2: Verify basis
    static_assert(Basis::order == amr::config::Order);
    static_assert(Basis::dimensions == amr::config::Dim);
    
    // Step 3: Verify globals
    static_assert(GC::volume_mass.rank() == amr::config::Dim);
    static_assert(GC::face_kernels.size() == 2);
    
    // Step 4: Verify equation
    static_assert(std::is_class_v<Adv>);
    
    // Step 5: Complex compile-time computation
    constexpr auto test_computation = []() {
        // All of this runs at compile-time
        double result = 0.0;
        for (unsigned int i = 0; i < Basis::quadweights.elements(); ++i) {
            result += Basis::quadweights[i];
        }
        return result;
    }();
    
    static_assert(test_computation > 0.99);
}
```

---

## Key Principles

1. **Everything is `constexpr`**: All components documented here are compile-time constants and functions. They can only be accessed in `constexpr` contexts (e.g., inside `static_assert` or `constexpr` variables).

2. **No Runtime Overhead**: Since evaluation happens at compile-time, there is **zero runtime overhead** for these computations.

3. **Type Safety**: The template-based design ensures type-safe specialization for different orders, dimensions, and equation types.

4. **Compile-Time Verification**: Use `static_assert` to verify properties at compile-time and catch configuration mismatches early.

5. **Reference Semantics**: Quadrature points and weights are accessed via `static constexpr const auto&` references to avoid copies.

---

## Compile-Time Guarantees

The test file `test_compile_time_config.cpp` demonstrates that **all** components pass compile-time verification:

```bash
# To run the compile-time test:
cd build
cmake --build . --target test_compile_time_config -j8

# If compilation succeeds (no errors), all static_asserts passed
# This proves everything is truly compile-time available
./bin/Debug/test_compile_time_config  # Exit code 0 confirms success
```

---

## References

- **Basis Module**: `include/dg_helpers/basis/`
  - `basis.hpp` - Main basis class
  - `polynomial.hpp` - Lagrange polynomial basis
  - `gauss_legendre.hpp` - Gauss-Legendre quadrature

- **Globals Module**: `include/dg_helpers/globals/`
  - `global_config.hpp` - GlobalConfig template
  - `kernels.hpp` - FaceKernels and MassTensors
  - `quadrature.hpp` - Quadrature data

- **Equations Module**: `include/dg_helpers/equations/`
  - `advection.hpp` - Advection equation
  - `equation_impl.hpp` - Base equation class

- **Containers**: `include/containers/`
  - `static_vector.hpp` - Fixed-size vector
  - `static_tensor.hpp` - Multi-dimensional array
  - `container_operations.hpp` - Operator overloads
