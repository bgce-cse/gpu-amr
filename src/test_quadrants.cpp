// Test case: 2D Riemann problem (quadrants)
// https://www.clawpack.org/gallery/pyclaw/gallery/quadrants.html

#include "EulerSolver2D.h"
#include <iostream>

int main() {
    // Grid parameters matching ClawPack example  
    int nx = 100;    
    int ny = 100;    
    double Lx = 1.0; 
    double Ly = 1.0; 
    
    // Create solver with CFL matching ClawPack
    EulerSolver2D solver(nx, ny, Lx, Ly, 1.4, 0.4);
    
    // Define quadrants initial condition
    auto quadrantsIC = [](double x, double y) -> std::vector<double> {
        std::vector<double> prim(4);
        
        // Define regions based on ClawPack
        bool right = (x >= 0.8);
        bool top = (y >= 0.8);
        
        // Set primitive variables based on quadrant
        if (right && top) {         // top-right
            prim[0] = 1.5;                // density
            prim[1] = 0.0;                // u velocity
            prim[2] = 0.0;                // v velocity
            prim[3] = 1.5;                // pressure
        } else if (!right && top) { // top-left
            prim[0] = 0.532258064516129;
            prim[1] = 1.206045378311055;
            prim[2] = 0.0;
            prim[3] = 0.3;
        } else if (!right && !top) { // bottom-left
            prim[0] = 0.137992831541219;
            prim[1] = 1.206045378311055;
            prim[2] = 1.206045378311055;
            prim[3] = 0.029032258064516;
        } else {                    // bottom-right
            prim[0] = 0.532258064516129;
            prim[1] = 0.0;
            prim[2] = 1.206045378311055;
            prim[3] = 0.3;
        }
        
        return prim;
    };
    
    // Initialize solver
    solver.initializeCustom(quadrantsIC);
    
    // Solve
    double tmax = 0.8;
    int output_interval = 10;
    solver.solve(tmax, output_interval, "quadrants");
    
    return 0;
}