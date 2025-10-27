// Test case: 2D Riemann problem (quadrants)
// https://www.clawpack.org/gallery/pyclaw/gallery/quadrants.html 

#include "EulerSolverND.h"
#include <iostream>

int main() {
    std::array<int, 2> n = {100, 100};
    std::array<double, 2> L = {1.0, 1.0};
    
    EulerSolver2D solver(n, L, 1.4, 0.4);
    
    auto quadrantsIC = [](const std::array<double, 2>& x) -> std::vector<double> {
        std::vector<double> prim(4);  // [rho, u, v, p]
        
        bool right = (x[0] >= 0.8);
        bool top = (x[1] >= 0.8);
        
        if (right && top) {         // Quadrant 1: (+,+) - High pressure
            prim = {1.5, 0.0, 0.0, 1.5};
        } else if (!right && top) { // Quadrant 2: (-,+) - Moving right
            prim = {0.532258064516129, 1.206045378311055, 0.0, 0.3};
        } else if (!right && !top) { // Quadrant 3: (-,-) - Low density, high velocity
            prim = {0.137992831541219, 1.206045378311055, 1.206045378311055, 0.029032258064516};
        } else {                    // Quadrant 4: (+,-) - Moving up
            prim = {0.532258064516129, 0.0, 1.206045378311055, 0.3};
        }
        
        return prim;
    };
    
    solver.initializeCustom(quadrantsIC);
    solver.solve(0.8, 10, "quadrants");
    
    return 0;
}
