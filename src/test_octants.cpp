// Test case: 3D Riemann problem (8 octants)

#include "EulerSolverND.h"
#include <iostream>

int main() {
    std::array<int, 3> n = {80, 80, 80};
    std::array<double, 3> L = {1.0, 1.0, 1.0};
    
    EulerSolver3D solver(n, L, 1.4, 0.3);
    
    auto octantsIC = [](const std::array<double, 3>& x) -> std::vector<double> {
        std::vector<double> prim(5);  // [rho, u, v, w, p]
        
        bool right = (x[0] >= 0.5);
        bool top = (x[1] >= 0.5);
        bool front = (x[2] >= 0.5);
        
        if (right && top && front) {         // Octant 1: (+,+,+) - High pressure
            prim = {1.5, 0.0, 0.0, 0.0, 1.5};
        } else if (!right && top && front) { // Octant 2: (-,+,+) - Moving right
            prim = {0.5323, 1.206, 0.0, 0.0, 0.3};
        } else if (!right && !top && front) { // Octant 3: (-,-,+) - Low density, high velocity
            prim = {0.138, 1.206, 1.206, 0.0, 0.029};
        } else if (right && !top && front) {  // Octant 4: (+,-,+) - Moving up
            prim = {0.5323, 0.0, 1.206, 0.0, 0.3};
        } else if (right && top && !front) {  // Octant 5: (+,+,-) - Moving forward
            prim = {0.5323, 0.0, 0.0, 1.206, 0.3};
        } else if (!right && top && !front) { // Octant 6: (-,+,-) - Moving right and forward
            prim = {0.8, 1.0, 0.0, 1.0, 0.4};
        } else if (!right && !top && !front) { // Octant 7: (-,-,-) - All directions
            prim = {0.5, 1.0, 1.0, 1.0, 0.2};
        } else {                              // Octant 8: (+,-,-) - Moving up and forward
            prim = {0.8, 0.0, 1.0, 1.0, 0.4};
        }
        
        return prim;
    };
    
    solver.initializeCustom(octantsIC);
    solver.solve(0.6, 5, "octants");
    
    return 0;
}
