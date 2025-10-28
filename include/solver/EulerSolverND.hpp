#ifndef EULER_SOLVER_ND_H
#define EULER_SOLVER_ND_H

#include <vector>
#include <string>
#include <functional>
#include <array>

// Helper template to create nested vector types
template<typename T, int N>
struct NestedVector {
    using type = std::vector<typename NestedVector<T, N-1>::type>;
};

// Helper template for nested vector types (1D, final for recursive call)
template<typename T>
struct NestedVector<T, 1> {
    using type = std::vector<T>;
};

template<int DIM>
class EulerSolverND {
private:
    std::array<int, DIM> n;           // Grid dimensions
    std::array<double, DIM> d;        // Grid spacing
    double gamma;                      // Specific heat ratio
    double cfl;                        // CFL number
    
    int nvar;                          // Number of variables (DIM + 2)
    
    // For 2D: std::vector<std::vector<std::vector<double>>>
    // For 3D: std::vector<std::vector<std::vector<std::vector<double>>>>
    typename NestedVector<std::vector<double>, DIM>::type U;
    typename NestedVector<std::vector<double>, DIM>::type U_new;
    
    void conservativeToPrimitive(const std::vector<double>& cons, std::vector<double>& prim) const;
    
    void primitiveToConservative(const std::vector<double>& prim, std::vector<double>& cons) const;
    
    void computeFlux(const std::vector<double>& cons, std::vector<double>& flux, int direction) const;
    
    double computeSoundSpeed(const std::vector<double>& cons) const;
    
    void rusanovFlux(const std::vector<double>& UL, const std::vector<double>& UR, 
                     std::vector<double>& flux, int direction) const;
    
    double computeTimeStep() const;
    
    void applyBoundaryConditions();
    
    // Initialize nested vector with given dimensions
    void initializeNestedVector(typename NestedVector<std::vector<double>, DIM>::type& vec, int dim_idx);
    
public:
    EulerSolverND(const std::array<int, DIM>& n_, 
                        const std::array<double, DIM>& L, 
                        double gamma_ = 1.4, 
                        double cfl_ = 0.3);
    
    void initializeCustom(std::function<std::vector<double>(const std::array<double, DIM>&)> init_func);
    
    void timeStep(double dt);
    
    void solve(double tmax, int output_interval = 1, const std::string& output_prefix = "solution");
    
    void writeToVTK(const std::string& filename) const;
    
    int getDim() const { return DIM; }
    int getNvar() const { return nvar; }
    const std::array<int, DIM>& getGridSize() const { return n; }
    const std::array<double, DIM>& getGridSpacing() const { return d; }
    
    const typename NestedVector<std::vector<double>, DIM>::type& getSolution() const { return U; }
};

// Typedefs
using EulerSolver1D = EulerSolverND<1>;
using EulerSolver2D = EulerSolverND<2>;
using EulerSolver3D = EulerSolverND<3>;

#endif // EULER_SOLVER_ND_H
