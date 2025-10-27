// Test case: 1D Sod shock tube

#include "EulerSolverND.h"
#include <iostream>

int main() {
    std::array<int, 1> n = {200};
    std::array<double, 1> L = {1.0};
    
    EulerSolver1D solver(n, L, 1.4, 0.4);
    
    auto sodIC = [](const std::array<double, 1>& x) -> std::vector<double> {
        std::vector<double> prim(3);  // [rho, u, p]
        
        if (x[0] < 0.5) {
            // Left state (high pressure)
            prim = {1.0, 0.0, 1.0};
        } else {
            // Right state (low pressure)
            prim = {0.125, 0.0, 0.1};
        }
        
        return prim;
    };
    
    solver.initializeCustom(sodIC);
    solver.solve(0.2, 10, "sod");
    
    return 0;
}
