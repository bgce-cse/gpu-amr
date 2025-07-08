// Compile with g++ -O2 -o quad_solver EulerFVM_Quad.cpp && ./quad_solver
// https://www.clawpack.org/gallery/pyclaw/gallery/quadrants.html

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm>

class EulerSolver2D {
private:
    int nx, ny;           // Grid dimensions
    double dx, dy;        // Grid spacing
    double gamma;         // Specific heat ratio
    double cfl;           // CFL number
    
    // Conservative variables: [rho, rho*u, rho*v, E]
    std::vector<std::vector<std::vector<double>>> U;
    std::vector<std::vector<std::vector<double>>> U_new;
    
public:
    EulerSolver2D(int nx_, int ny_, double Lx, double Ly, double gamma_ = 1.4, double cfl_ = 0.3)
        : nx(nx_), ny(ny_), gamma(gamma_), cfl(cfl_) {
        dx = Lx / (nx - 1);
        dy = Ly / (ny - 1);
        
        // Initialize arrays: U[i][j][var] where var = 0,1,2,3 for rho,rhou,rhov,E
        U.resize(nx, std::vector<std::vector<double>>(ny, std::vector<double>(4, 0.0)));
        U_new.resize(nx, std::vector<std::vector<double>>(ny, std::vector<double>(4, 0.0)));
    }
    
    // Convert conservative to primitive variables
    void conservativeToPrimitive(const std::vector<double>& cons, std::vector<double>& prim) {
        double rho = cons[0];
        double u = cons[1] / rho;
        double v = cons[2] / rho;
        double E = cons[3];
        double p = (gamma - 1.0) * (E - 0.5 * rho * (u*u + v*v));
        
        prim[0] = rho;  // density
        prim[1] = u;    // x-velocity
        prim[2] = v;    // y-velocity
        prim[3] = p;    // pressure
    }
    
    // Convert primitive to conservative variables
    void primitiveToConservative(const std::vector<double>& prim, std::vector<double>& cons) {
        double rho = prim[0];
        double u = prim[1];
        double v = prim[2];
        double p = prim[3];
        
        cons[0] = rho;                                          // rho
        cons[1] = rho * u;                                      // rho*u
        cons[2] = rho * v;                                      // rho*v
        cons[3] = p/(gamma-1.0) + 0.5*rho*(u*u + v*v);         // E
    }
    
    // Compute flux in x-direction
    void computeFluxX(const std::vector<double>& cons, std::vector<double>& flux) {
        std::vector<double> prim(4);
        conservativeToPrimitive(cons, prim);
        
        double rho = prim[0];
        double u = prim[1];
        double v = prim[2];
        double p = prim[3];
        
        flux[0] = rho * u;                      // rho*u
        flux[1] = rho * u * u + p;              // rho*u^2 + p
        flux[2] = rho * u * v;                  // rho*u*v
        flux[3] = u * (cons[3] + p);            // u*(E + p)
    }
    
    // Compute flux in y-direction
    void computeFluxY(const std::vector<double>& cons, std::vector<double>& flux) {
        std::vector<double> prim(4);
        conservativeToPrimitive(cons, prim);
        
        double rho = prim[0];
        double u = prim[1];
        double v = prim[2];
        double p = prim[3];
        
        flux[0] = rho * v;                      // rho*v
        flux[1] = rho * u * v;                  // rho*u*v
        flux[2] = rho * v * v + p;              // rho*v^2 + p
        flux[3] = v * (cons[3] + p);            // v*(E + p)
    }
    
    // Compute sound speed
    double computeSoundSpeed(const std::vector<double>& cons) {
        std::vector<double> prim(4);
        conservativeToPrimitive(cons, prim);
        double rho = prim[0];
        double p = prim[3];
        return sqrt(gamma * p / rho);
    }
    
    // Rusanov (Local Lax-Friedrichs) Riemann solver for x-direction
    void rusanovFluxX(const std::vector<double>& UL, const std::vector<double>& UR, 
                      std::vector<double>& flux) {
        std::vector<double> fluxL(4), fluxR(4);
        computeFluxX(UL, fluxL);
        computeFluxX(UR, fluxR);
        
        std::vector<double> primL(4), primR(4);
        conservativeToPrimitive(UL, primL);
        conservativeToPrimitive(UR, primR);
        
        double aL = computeSoundSpeed(UL);
        double aR = computeSoundSpeed(UR);
        double uL = primL[1];
        double uR = primR[1];
        
        // Maximum wave speed
        double smax = std::max(std::abs(uL) + aL, std::abs(uR) + aR);
        
        // Rusanov flux
        for (int k = 0; k < 4; k++) {
            flux[k] = 0.5 * (fluxL[k] + fluxR[k]) - 0.5 * smax * (UR[k] - UL[k]);
        }
    }
    
    // Rusanov (Local Lax-Friedrichs) Riemann solver for y-direction
    void rusanovFluxY(const std::vector<double>& UL, const std::vector<double>& UR, 
                      std::vector<double>& flux) {
        std::vector<double> fluxL(4), fluxR(4);
        computeFluxY(UL, fluxL);
        computeFluxY(UR, fluxR);
        
        std::vector<double> primL(4), primR(4);
        conservativeToPrimitive(UL, primL);
        conservativeToPrimitive(UR, primR);
        
        double aL = computeSoundSpeed(UL);
        double aR = computeSoundSpeed(UR);
        double vL = primL[2];
        double vR = primR[2];
        
        // Maximum wave speed
        double smax = std::max(std::abs(vL) + aL, std::abs(vR) + aR);
        
        // Rusanov flux
        for (int k = 0; k < 4; k++) {
            flux[k] = 0.5 * (fluxL[k] + fluxR[k]) - 0.5 * smax * (UR[k] - UL[k]);
        }
    }
    
    // Compute time step based on CFL condition
    double computeTimeStep() {
        double dt_min = 1e10;
        
        for (int i = 1; i < nx-1; i++) {
            for (int j = 1; j < ny-1; j++) {
                std::vector<double> prim(4);
                conservativeToPrimitive(U[i][j], prim);
                
                double u = prim[1];
                double v = prim[2];
                double a = computeSoundSpeed(U[i][j]);
                
                double dt_x = dx / (std::abs(u) + a);
                double dt_y = dy / (std::abs(v) + a);
                
                dt_min = std::min(dt_min, std::min(dt_x, dt_y));
            }
        }
        
        return cfl * dt_min;
    }
    
    // Apply boundary conditions (extrapolation - matching ClawPack)
    void applyBoundaryConditions() {
        // Left and right boundaries - extrapolation
        for (int j = 0; j < ny; j++) {
            // Left boundary - extrapolate from interior
            U[0][j] = U[1][j];
            
            // Right boundary - extrapolate from interior
            U[nx-1][j] = U[nx-2][j];
        }
        
        // Top and bottom boundaries - extrapolation
        for (int i = 0; i < nx; i++) {
            // Bottom boundary - extrapolate from interior
            U[i][0] = U[i][1];
            
            // Top boundary - extrapolate from interior
            U[i][ny-1] = U[i][ny-2];
        }
    }
    
    // Initialize 2D Riemann problem (Liska-Wendroff quadrants case)
    void initializeQuadrants() {
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                double x = i * dx;
                double y = j * dy;
                std::vector<double> prim(4);
                
                // Define regions based on ClawPack convention
                bool l = (x < 0.8);  // left
                bool r = (x >= 0.8); // right
                bool b = (y < 0.8);  // bottom
                bool t = (y >= 0.8); // top
                
                // Set density (matching ClawPack exactly)
                if (r && t) {
                    prim[0] = 1.5;                // top-right
                } else if (l && t) {
                    prim[0] = 0.532258064516129;  // top-left
                } else if (l && b) {
                    prim[0] = 0.137992831541219;  // bottom-left
                } else { // r && b
                    prim[0] = 0.532258064516129;  // bottom-right
                }
                
                // Set u velocity
                if (r && t) {
                    prim[1] = 0.0;
                } else if (l && t) {
                    prim[1] = 1.206045378311055;
                } else if (l && b) {
                    prim[1] = 1.206045378311055;
                } else { // r && b
                    prim[1] = 0.0;
                }
                
                // Set v velocity
                if (r && t) {
                    prim[2] = 0.0;
                } else if (l && t) {
                    prim[2] = 0.0;
                } else if (l && b) {
                    prim[2] = 1.206045378311055;
                } else { // r && b
                    prim[2] = 1.206045378311055;
                }
                
                // Set pressure
                if (r && t) {
                    prim[3] = 1.5;
                } else if (l && t) {
                    prim[3] = 0.3;
                } else if (l && b) {
                    prim[3] = 0.029032258064516;
                } else { // r && b
                    prim[3] = 0.3;
                }
                
                primitiveToConservative(prim, U[i][j]);
            }
        }
    }
    
    // Single time step using explicit Euler method
    void timeStep(double dt) {
        // Copy current solution
        U_new = U;
        
        // Compute fluxes and update interior cells
        for (int i = 1; i < nx-1; i++) {
            for (int j = 1; j < ny-1; j++) {
                std::vector<double> fluxL(4), fluxR(4), fluxB(4), fluxT(4);
                
                // X-direction fluxes
                rusanovFluxX(U[i][j], U[i+1][j], fluxR);    // Right face
                rusanovFluxX(U[i-1][j], U[i][j], fluxL);    // Left face
                
                // Y-direction fluxes
                rusanovFluxY(U[i][j], U[i][j+1], fluxT);    // Top face
                rusanovFluxY(U[i][j-1], U[i][j], fluxB);    // Bottom face
                
                // Update conservative variables
                for (int k = 0; k < 4; k++) {
                    U_new[i][j][k] = U[i][j][k] - dt/dx * (fluxR[k] - fluxL[k]) 
                                                 - dt/dy * (fluxT[k] - fluxB[k]);
                }
            }
        }
        
        // Update solution
        U = U_new;
        
        // Apply boundary conditions
        applyBoundaryConditions();
    }
    
    // Write solution to VTK file for ParaView
    void writeToVTK(const std::string& filename) {
        std::ofstream file(filename);
        
        // VTK header
        file << "# vtk DataFile Version 3.0" << std::endl;
        file << "2D Euler Quadrants" << std::endl;
        file << "ASCII" << std::endl;
        file << "DATASET STRUCTURED_GRID" << std::endl;
        
        // Grid dimensions
        file << "DIMENSIONS " << nx << " " << ny << " 1" << std::endl;
        
        // Points (grid coordinates)
        file << "POINTS " << nx * ny << " float" << std::endl;
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                double x = i * dx;
                double y = j * dy;
                file << x << " " << y << " 0.0" << std::endl;
            }
        }
        
        // Point data (values at grid points)
        file << "POINT_DATA " << nx * ny << std::endl;
        
        // Density
        file << "SCALARS density float 1" << std::endl;
        file << "LOOKUP_TABLE default" << std::endl;
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                std::vector<double> prim(4);
                conservativeToPrimitive(U[i][j], prim);
                file << prim[0] << std::endl;
            }
        }
        
        // Pressure
        file << "SCALARS pressure float 1" << std::endl;
        file << "LOOKUP_TABLE default" << std::endl;
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                std::vector<double> prim(4);
                conservativeToPrimitive(U[i][j], prim);
                file << prim[3] << std::endl;
            }
        }
        
        // Velocity magnitude
        file << "SCALARS velocity_magnitude float 1" << std::endl;
        file << "LOOKUP_TABLE default" << std::endl;
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                std::vector<double> prim(4);
                conservativeToPrimitive(U[i][j], prim);
                double vel_mag = sqrt(prim[1]*prim[1] + prim[2]*prim[2]);
                file << vel_mag << std::endl;
            }
        }
        
        // Velocity vector
        file << "VECTORS velocity float" << std::endl;
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                std::vector<double> prim(4);
                conservativeToPrimitive(U[i][j], prim);
                file << prim[1] << " " << prim[2] << " 0.0" << std::endl;
            }
        }
        
        file.close();
    }
    
    // Main solver loop
    void solve(double tmax, int output_interval = 1) {
        double t = 0.0;
        int step = 0;
        int file_counter = 0;
        
        std::cout << "Starting 2D Euler quadrants simulation..." << std::endl;
        std::cout << "Grid: " << nx << " x " << ny << std::endl;
        
        // Create output directory
        #ifdef _WIN32
            system("mkdir output 2>nul");
        #else
            system("mkdir -p output");
        #endif
        
        // Write initial condition
        char filename[100];
        sprintf(filename, "output/quadrants_%06d.vtk", file_counter);
        writeToVTK(filename);
        file_counter++;
        
        while (t < tmax) {
            double dt = computeTimeStep();
            
            if (t + dt > tmax) {
                dt = tmax - t;
            }
            
            timeStep(dt);
            
            t += dt;
            step++;
            
            // Output every output_interval steps
            if (step % output_interval == 0) {
                sprintf(filename, "output/quadrants_%06d.vtk", file_counter);
                writeToVTK(filename);
                file_counter++;
                
                std::cout << "Step: " << step << ", Time: " << t << std::endl;
            }
            
            // Safety break
            if (step > 50000) {
                std::cout << "Breaking after 50000 steps for safety..." << std::endl;
                break;
            }
        }
        
        // Write final solution
        sprintf(filename, "output/quadrants_%06d.vtk", file_counter);
        writeToVTK(filename);
        
        std::cout << "Simulation completed. Files in output/ directory." << std::endl;
    }
};

int main() {
    // Grid parameters matching ClawPack example  
    int nx = 100;    
    int ny = 100;    
    double Lx = 1.0; 
    double Ly = 1.0; 
    
    // Create solver with CFL matching ClawPack
    EulerSolver2D solver(nx, ny, Lx, Ly, 1.4, 0.4);
    
    // Initialize quadrants problem
    solver.initializeQuadrants();
    
    // Solve with parameters matching ClawPack
    double tmax = 0.8;
    int output_interval = 10;
    solver.solve(tmax, output_interval);
    
    return 0;
}