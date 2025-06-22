#include "../include/dg_helpers/basis.hpp"
#include "../include/dg_helpers/globals.hpp"
#include <iostream>
#include <cmath>
#include <cassert>
#include <functional>

using namespace amr::basis;
using namespace amr::globals;

template <unsigned int Order>
void test_evaluate_basis() {
    // Create Basis object for 3D, with GaussLegendre nodes on [0,1]
    Basis<3, Order> basis(0.0, 1.0);

    // Access GaussLegendre object inside Basis
    const auto& pts = basis.getGL().points();

    // Use the correct coeff_matrix_t type from Basis
    typename Basis<3, Order>::coeff_matrix_t coeffs{};

    // Fill coefficients for the function f(x,y,z) = x + y + z at quadrature nodes
    for (std::size_t k = 0; k < Order; ++k) {
        for (std::size_t j = 0; j < Order; ++j) {
            for (std::size_t i = 0; i < Order; ++i) {
                std::size_t index = i + Order * (j + Order * k);
                double x = pts[i];
                double y = pts[j];
                double z = pts[k];
                coeffs[index] = x + y + z;
            }
        }
    }

    // Test points where we evaluate the interpolant
    std::array<coord3d, 3> test_points = {{
        {0.25, 0.25, 0.25},
        {0.5, 0.5, 0.5},
        {0.75, 0.75, 0.75}
    }};

    for (const auto& v : test_points) {
        double expected = v[0] + v[1] + v[2];
        double interp = basis.evaluate_basis(coeffs, v);
        double error = std::abs(interp - expected);

        std::cout << "At (" << v[0] << ", " << v[1] << ", " << v[2] << "): "
                  << "Expected = " << expected << ", Interpolated = " << interp
                  << ", Error = " << error << '\n';

        assert(error < 1e-4);
    }

    std::cout << "evaluate_basis test passed for Order = " << Order << std::endl;
}
void test_globals() {
    Globals gm;

    std::array<Face, 6> faces = {
        Face::Left, Face::Right,
        Face::Bottom, Face::Top,
        Face::Back, Face::Front
    };

    for (Face f : faces) {
        Face opp = gm.opposite_faces[f];

        std::cout << "Testing face " << to_string(f) << '\n';
        std::cout << "  Opposite: " << to_string(opp) << '\n';
        std::cout << "  Normal Sign: " << gm.normal_signs[f] << '\n';
        std::cout << "  Axis Index: " << gm.normal_idxs[f] << '\n';

        // Round-trip test
        assert(gm.opposite_faces[opp] == f);

        // Normal sign should flip for opposite face
        assert(gm.normal_signs[f] == -gm.normal_signs[opp]);
    }
    Basis<3, 5> basis(0.0, 1.0);
    gm.quadweights_nd = basis.mass_matrix_diagonal(3);
    std::cout<< gm.quadweights_nd.size() << "\n";
    std::cout << "Globals test passed.\n";
}

int main() {
    test_evaluate_basis<3>();
    test_evaluate_basis<4>();
    test_globals();
    return 0;
}