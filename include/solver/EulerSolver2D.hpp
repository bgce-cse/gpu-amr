#ifndef EULER_SOLVER_2D_H
#define EULER_SOLVER_2D_H

#include <vector>
#include <string>
#include <functional>

namespace Direction {
    constexpr int X = 0;
    constexpr int Y = 1;
}

class EulerSolver2D {
private:
    int nx, ny;           // Grid dimensions
    double dx, dy;        // Grid spacing
    double gamma;         // Specific heat ratio
    double cfl;           // CFL number
    
    std::vector<std::vector<std::vector<double>>> U;
    std::vector<std::vector<std::vector<double>>> U_new;
    
public:
    EulerSolver2D(int nx_, int ny_, double Lx, double Ly, double gamma_ = 1.4, double cfl_ = 0.3)
        : nx(nx_), ny(ny_), gamma(gamma_), cfl(cfl_) {
        dx = Lx / (nx - 1);
        dy = Ly / (ny - 1);
        
        // Initialize arrays: U[i][j][var] (var = 0,1,2,3 for rho,rhou,rhov,E)
        U.resize(nx, std::vector<std::vector<double>>(ny, std::vector<double>(4, 0.0)));
        U_new.resize(nx, std::vector<std::vector<double>>(ny, std::vector<double>(4, 0.0)));
    }
    
    void initializeCustom(std::function<std::vector<double>(double, double)> init_func){
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                double x = i * dx;
                double y = j * dy;
                std::vector<double> prim = init_func(x, y);
                primitiveToConservative(prim, U[i][j]);
            }
        }
    }
    
    void timeStep(double dt){
        // Copy current solution
        U_new = U;
        
        // Compute fluxes and update interior cells
        for (int i = 1; i < nx-1; i++) {
            for (int j = 1; j < ny-1; j++) {
                std::vector<double> fluxL(4), fluxR(4), fluxB(4), fluxT(4);
                
                // X-direction fluxes
                rusanovFlux(U[i][j], U[i+1][j], fluxR, Direction::X);    // Right face
                rusanovFlux(U[i-1][j], U[i][j], fluxL, Direction::X);    // Left face
                
                // Y-direction fluxes
                rusanovFlux(U[i][j], U[i][j+1], fluxT, Direction::Y);    // Top face
                rusanovFlux(U[i][j-1], U[i][j], fluxB, Direction::Y);    // Bottom face
                
                // Update conservative variables using finite volume formula
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
    
    void solve(double tmax, int output_interval = 1, const std::string& output_prefix = "solution"){
        double t = 0.0;
        int step = 0;
        int file_counter = 0;
        
        std::cout << "Starting 2D Euler simulation...\n";
        std::cout << "Grid: " << nx << " x " << ny << "\n";
        
        // Create output directory
        #ifdef _WIN32
            system("mkdir output 2>nul");
        #else
            system("mkdir -p ../output");
        #endif
        
        // Write initial condition
        char filename[200];
        sprintf(filename, "../output/%s_%06d.vtk", output_prefix.c_str(), file_counter);
        writeToVTK(filename);
        file_counter++;
        
        // Main time-stepping loop
        while (t < tmax) {
            double dt = computeTimeStep();
            
            if (t + dt > tmax) {
                dt = tmax - t;  // Adjust final time step
            }
            
            timeStep(dt);
            
            t += dt;
            step++;
            
            // Output every output_interval steps
            if (step % output_interval == 0) {
                sprintf(filename, "../output/%s_%06d.vtk", output_prefix.c_str(), file_counter);
                writeToVTK(filename);
                file_counter++;
                
                std::cout << "Step: " << step << ", Time: " << t << std::endl;
            }
            
            // Safety break to prevent infinite loops
            if (step > 50000) {
                std::cout << "Breaking after 50000 steps for safety...\n";
                break;
            }
        }
        
        // Write final solution
        sprintf(filename, "../output/%s_%06d.vtk", output_prefix.c_str(), file_counter);
        writeToVTK(filename);
        
        std::cout << "Simulation completed. Files in output/ directory.\n";
    }

    void conservativeToPrimitive(const std::vector<double>& cons, std::vector<double>& prim) const{
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
    
    void primitiveToConservative(const std::vector<double>& prim, std::vector<double>& cons) const{
        double rho = prim[0];
        double u = prim[1];
        double v = prim[2];
        double p = prim[3];
        
        cons[0] = rho;                                          // rho
        cons[1] = rho * u;                                      // rho*u
        cons[2] = rho * v;                                      // rho*v
        cons[3] = p/(gamma-1.0) + 0.5*rho*(u*u + v*v);         // E
    }
    
    void computeFlux(const std::vector<double>& cons, std::vector<double>& flux, int direction) const{
        std::vector<double> prim(4);
        conservativeToPrimitive(cons, prim);
        
        double rho = prim[0];
        double u = prim[1];
        double v = prim[2];
        double p = prim[3];
        
        if (direction == Direction::X) {
            // F = [rho*u, rho*u^2 + p, rho*u*v, u*(E + p)]
            flux[0] = rho * u;                      // rho*u
            flux[1] = rho * u * u + p;              // rho*u^2 + p
            flux[2] = rho * u * v;                  // rho*u*v
            flux[3] = u * (cons[3] + p);            // u*(E + p)
        } else { // Direction::Y
            // G = [rho*v, rho*u*v, rho*v^2 + p, v*(E + p)]
            flux[0] = rho * v;                      // rho*v
            flux[1] = rho * u * v;                  // rho*u*v
            flux[2] = rho * v * v + p;              // rho*v^2 + p
            flux[3] = v * (cons[3] + p);            // v*(E + p)
        }
    }
    
    double computeSoundSpeed(const std::vector<double>& cons) const{
        std::vector<double> prim(4);
        conservativeToPrimitive(cons, prim);
        double rho = prim[0];
        double p = prim[3];
        return sqrt(gamma * p / rho);
    }
    
    void rusanovFlux(const std::vector<double>& UL, const std::vector<double>& UR, 
                     std::vector<double>& flux, int direction) const{
        std::vector<double> fluxL(4), fluxR(4);
        computeFlux(UL, fluxL, direction);
        computeFlux(UR, fluxR, direction);
        
        std::vector<double> primL(4), primR(4);
        conservativeToPrimitive(UL, primL);
        conservativeToPrimitive(UR, primR);
        
        double aL = computeSoundSpeed(UL);
        double aR = computeSoundSpeed(UR);
        
        // Maximum wave speed based on direction
        double smax;
        if (direction == Direction::X) {
            double uL = primL[1];
            double uR = primR[1];
            smax = std::max(std::abs(uL) + aL, std::abs(uR) + aR);
        } else { // Direction::Y
            double vL = primL[2];
            double vR = primR[2];
            smax = std::max(std::abs(vL) + aL, std::abs(vR) + aR);
        }
        
        // Rusanov flux: F* = 0.5*(FL + FR) - 0.5*smax*(UR - UL)
        for (int k = 0; k < 4; k++) {
            flux[k] = 0.5 * (fluxL[k] + fluxR[k]) - 0.5 * smax * (UR[k] - UL[k]);
        }
    }
    
    double computeTimeStep() const{
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
    
    void applyBoundaryConditions(){
        // Left and right boundaries
        for (int j = 0; j < ny; j++) {
            U[0][j] = U[1][j];                  // Left boundary
            U[nx-1][j] = U[nx-2][j];            // Right boundary
        }
        
        // Top and bottom boundaries
        for (int i = 0; i < nx; i++) {
            U[i][0] = U[i][1];                  // Bottom boundary
            U[i][ny-1] = U[i][ny-2];            // Top boundary
        }
    }
    
    void writeToVTK(const std::string& filename) const{
        std::ofstream file(filename);
        
        // VTK header
        file << "# vtk DataFile Version 3.0\n";
        file << "2D Euler Solution\n";
        file << "ASCII\n";
        file << "DATASET STRUCTURED_GRID\n";
        
        // Grid dimensions
        file << "DIMENSIONS " << nx << " " << ny << " 1\n";
        
        // Points (grid coordinates)
        file << "POINTS " << nx * ny << " float\n";
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                double x = i * dx;
                double y = j * dy;
                file << x << " " << y << " 0.0\n";
            }
        }
        
        // Point data (values at grid points)
        file << "POINT_DATA " << nx * ny << "\n";
        
        // Density
        file << "SCALARS density float 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                std::vector<double> prim(4);
                conservativeToPrimitive(U[i][j], prim);
                file << prim[0] << "\n";
            }
        }
        
        // Pressure
        file << "SCALARS pressure float 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                std::vector<double> prim(4);
                conservativeToPrimitive(U[i][j], prim);
                file << prim[3] << "\n";
            }
        }
        
        // Velocity magnitude
        file << "SCALARS velocity_magnitude float 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                std::vector<double> prim(4);
                conservativeToPrimitive(U[i][j], prim);
                double vel_mag = sqrt(prim[1]*prim[1] + prim[2]*prim[2]);
                file << vel_mag << "\n";
            }
        }
        
        // Velocity vector
        file << "VECTORS velocity float\n";
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                std::vector<double> prim(4);
                conservativeToPrimitive(U[i][j], prim);
                file << prim[1] << " " << prim[2] << " 0.0\n";
            }
        }
        
        file.close();
    }
    
    int getNx() const { return nx; }
    int getNy() const { return ny; }
    double getDx() const { return dx; }
    double getDy() const { return dy; }
    
    const std::vector<std::vector<std::vector<double>>>& getSolution() const { return U; }
};



#endif // EULER_SOLVER_2D_H