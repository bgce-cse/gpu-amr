#ifndef EULER_SOLVER_2D_H
#define EULER_SOLVER_2D_H

#include <vector>
#include <string>
#include <functional>

class EulerSolver2D {
private:
    int nx, ny;           // Grid dimensions
    double dx, dy;        // Grid spacing
    double gamma;         // Specific heat ratio
    double cfl;           // CFL number
    
    std::vector<std::vector<std::vector<double>>> U;
    std::vector<std::vector<std::vector<double>>> U_new;
    
    void conservativeToPrimitive(const std::vector<double>& cons, std::vector<double>& prim) const;
    
    void primitiveToConservative(const std::vector<double>& prim, std::vector<double>& cons) const;
    
    void computeFlux(const std::vector<double>& cons, std::vector<double>& flux, int direction) const;
    
    double computeSoundSpeed(const std::vector<double>& cons) const;
    
    void rusanovFlux(const std::vector<double>& UL, const std::vector<double>& UR, 
                     std::vector<double>& flux, int direction) const;
    
    double computeTimeStep() const;
    
    void applyBoundaryConditions();
    
public:
    EulerSolver2D(int nx_, int ny_, double Lx, double Ly, double gamma_ = 1.4, double cfl_ = 0.3);
    
    void initializeCustom(std::function<std::vector<double>(double, double)> init_func);
    
    void timeStep(double dt);
    
    void solve(double tmax, int output_interval = 1, const std::string& output_prefix = "solution");
    
    void writeToVTK(const std::string& filename) const;
    
    int getNx() const { return nx; }
    int getNy() const { return ny; }
    double getDx() const { return dx; }
    double getDy() const { return dy; }
    
    const std::vector<std::vector<std::vector<double>>>& getSolution() const { return U; }
};

namespace Direction {
    constexpr int X = 0;
    constexpr int Y = 1;
}

#endif // EULER_SOLVER_2D_H