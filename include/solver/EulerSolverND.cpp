// Supports 1D, 2D, and 3D
// Compile with: g++ -O2 -std=c++17 -o test_[FILE] test_[FILE].cpp EulerSolverND.cpp

#include "EulerSolverND.h"
#include <iostream>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <functional>
#include <cstdio>
#include <vector>
#include <string>
#include <cstdlib>

template<int DIM>
EulerSolverND<DIM>::EulerSolverND(const std::array<int, DIM>& n_, const std::array<double, DIM>& L, double gamma_, double cfl_)
    : n(n_), gamma(gamma_), cfl(cfl_), nvar(DIM + 2) {
    
    // Calculate grid spacing for each dimension
    for (int dim = 0; dim < DIM; dim++) {
        d[dim] = L[dim] / (n[dim] - 1);
    }
    
    // Initialize nested vector structure
    initializeNestedVector(U, 0);
    initializeNestedVector(U_new, 0);
}

// Initialize nested vectors - dimension specific
template<int DIM>
void EulerSolverND<DIM>::initializeNestedVector(
    typename NestedVector<std::vector<double>, DIM>::type& vec, int dim_idx) {
    
    if constexpr (DIM == 1) {
        // 1D: vector<vector<double>>
        vec.resize(n[0]);
        for (auto& v : vec) {
            v.resize(nvar, 0.0);
        }
    } else if constexpr (DIM == 2) {
        // 2D: vector<vector<vector<double>>>
        vec.resize(n[0]);
        for (auto& v : vec) {
            v.resize(n[1]);
            for (auto& vv : v) {
                vv.resize(nvar, 0.0);
            }
        }
    } else if constexpr (DIM == 3) {
        // 3D: vector<vector<vector<vector<double>>>>
        vec.resize(n[0]);
        for (auto& v : vec) {
            v.resize(n[1]);
            for (auto& vv : v) {
                vv.resize(n[2]);
                for (auto& vvv : vv) {
                    vvv.resize(nvar, 0.0);
                }
            }
        }
    }
}

// Convert conservative variables [rho, rho*u, rho*v, E] to primitive [rho, u, v, p]
template<int DIM>
void EulerSolverND<DIM>::conservativeToPrimitive(const std::vector<double>& cons, std::vector<double>& prim) const {
    double rho = cons[0];
    
    // Density (rho)
    prim[0] = rho;

    // Velocities (u)
    double vel_squared = 0.0;
    for (int dim = 0; dim < DIM; dim++) {
        prim[1 + dim] = cons[1 + dim] / rho;
        vel_squared += prim[1 + dim] * prim[1 + dim];
    }
    
    // Pressure (p)
    double E = cons[DIM + 1];
    prim[DIM + 1] = (gamma - 1.0) * (E - 0.5 * rho * vel_squared);
}

// Convert primitive variables [rho, u, v, p] to conservative [rho, rho*u, rho*v, E]
template<int DIM>
void EulerSolverND<DIM>::primitiveToConservative(const std::vector<double>& prim, std::vector<double>& cons) const {
    double rho = prim[0];
    double p = prim[DIM + 1];
    
    // Density (rho)
    cons[0] = rho;
    
    // Momentum and compute velocity squared (rho*u)
    double vel_squared = 0.0;
    for (int dim = 0; dim < DIM; dim++) {
        double u = prim[1 + dim];
        cons[1 + dim] = rho * u;
        vel_squared += u * u;
    }
    
    // Total energy (E)
    cons[DIM + 1] = p / (gamma - 1.0) + 0.5 * rho * vel_squared;
}

// Compute flux vectors from conservative variables
template<int DIM>
void EulerSolverND<DIM>::computeFlux(const std::vector<double>& cons, std::vector<double>& flux, int direction) const {
    std::vector<double> prim(nvar);
    conservativeToPrimitive(cons, prim);
    
    double rho = prim[0];
    double p = prim[DIM + 1];
    double u_dir = prim[1 + direction];
    
    // Mass flux (rho*u)
    flux[0] = rho * u_dir;
    
    // Momentum flux (rho*u*v, if direction rho*u^2 + p)
    for (int dim = 0; dim < DIM; dim++) {
        flux[1 + dim] = rho * u_dir * prim[1 + dim];
        if (dim == direction) { 
            flux[1 + dim] += p;
        }
    }
    
    // Energy flux (u*(E + p))
    flux[DIM + 1] = u_dir * (cons[DIM + 1] + p);
}

// Calculate sound speed
template<int DIM>
double EulerSolverND<DIM>::computeSoundSpeed(const std::vector<double>& cons) const {
    std::vector<double> prim(nvar);
    conservativeToPrimitive(cons, prim);
    double rho = prim[0];
    double p = prim[DIM + 1];
    return sqrt(gamma * p / rho);
}

// Rusanov (Local Lax-Friedrichs) flux - solves Riemann problem at cell interfaces
template<int DIM>
void EulerSolverND<DIM>::rusanovFlux(const std::vector<double>& UL,  const std::vector<double>& UR,  std::vector<double>& flux, int direction) const {
    std::vector<double> fluxL(nvar), fluxR(nvar);
    computeFlux(UL, fluxL, direction);
    computeFlux(UR, fluxR, direction);
    
    std::vector<double> primL(nvar), primR(nvar);
    conservativeToPrimitive(UL, primL);
    conservativeToPrimitive(UR, primR);
    
    double aL = computeSoundSpeed(UL);
    double aR = computeSoundSpeed(UR);
    
    double uL = primL[1 + direction];
    double uR = primR[1 + direction];

    // Maximum wave speed based on direction
    double smax = std::max(std::abs(uL) + aL, std::abs(uR) + aR);
    
    // Rusanov flux: F* = 0.5*(FL + FR) - 0.5*smax*(UR - UL)
    for (int k = 0; k < nvar; k++) {
        flux[k] = 0.5 * (fluxL[k] + fluxR[k]) - 0.5 * smax * (UR[k] - UL[k]);
    }
}

// Calculate maximum stable time step based on CFL condition
template<int DIM>
double EulerSolverND<DIM>::computeTimeStep() const {
    double dt_min = 1e10;
    
    if constexpr (DIM == 1) {
        for (int i = 1; i < n[0]-1; i++) {
            std::vector<double> prim(nvar);
            conservativeToPrimitive(U[i], prim);

            double u = prim[1];
            double a = computeSoundSpeed(U[i]);

            double dt_x = d[0] / (std::abs(u) + a);
            dt_min = std::min(dt_min, dt_x);
        }

    } else if constexpr (DIM == 2) {
        for (int i = 1; i < n[0]-1; i++) {
            for (int j = 1; j < n[1]-1; j++) {
                std::vector<double> prim(nvar);
                conservativeToPrimitive(U[i][j], prim);

                double a = computeSoundSpeed(U[i][j]);

                for (int dim = 0; dim < DIM; dim++) {
                    double u = prim[1 + dim];
                    double dt_dir = d[dim] / (std::abs(u) + a);
                    dt_min = std::min(dt_min, dt_dir);
                }
            }
        }

    } else if constexpr (DIM == 3) {
        for (int i = 1; i < n[0]-1; i++) {
            for (int j = 1; j < n[1]-1; j++) {
                for (int k = 1; k < n[2]-1; k++) {
                    std::vector<double> prim(nvar);
                    conservativeToPrimitive(U[i][j][k], prim);

                    double a = computeSoundSpeed(U[i][j][k]);

                    for (int dim = 0; dim < DIM; dim++) {
                        double u = prim[1 + dim];
                        double dt_dir = d[dim] / (std::abs(u) + a);
                        dt_min = std::min(dt_min, dt_dir);
                    }
                }
            }
        }
    }
    
    return cfl * dt_min;
}

// Apply boundary conditions (Neumann with 0 gradient)
template<int DIM>
void EulerSolverND<DIM>::applyBoundaryConditions() {
    if constexpr (DIM == 1) {
        U[0] = U[1];
        U[n[0]-1] = U[n[0]-2];

    } else if constexpr (DIM == 2) {
        // Left and right boundaries
        for (int j = 0; j < n[1]; j++) {
            U[0][j] = U[1][j];
            U[n[0]-1][j] = U[n[0]-2][j];
        }

        // Top and bottom boundaries
        for (int i = 0; i < n[0]; i++) {
            U[i][0] = U[i][1];
            U[i][n[1]-1] = U[i][n[1]-2];
        }

    } else if constexpr (DIM == 3) {
        // X boundaries
        for (int j = 0; j < n[1]; j++) {
            for (int k = 0; k < n[2]; k++) {
                U[0][j][k] = U[1][j][k];
                U[n[0]-1][j][k] = U[n[0]-2][j][k];
            }
        }

        // Y boundaries
        for (int i = 0; i < n[0]; i++) {
            for (int k = 0; k < n[2]; k++) {
                U[i][0][k] = U[i][1][k];
                U[i][n[1]-1][k] = U[i][n[1]-2][k];
            }
        }

        // Z boundaries
        for (int i = 0; i < n[0]; i++) {
            for (int j = 0; j < n[1]; j++) {
                U[i][j][0] = U[i][j][1];
                U[i][j][n[2]-1] = U[i][j][n[2]-2];
            }
        }
    }
}

// Initialize domain using custom function that returns [rho, u, v, p] at each (x,y)
template<int DIM>
void EulerSolverND<DIM>::initializeCustom(
    std::function<std::vector<double>(const std::array<double, DIM>&)> init_func) {
    
    if constexpr (DIM == 1) {
        for (int i = 0; i < n[0]; i++) {
            std::array<double, 1> x = {i * d[0]};
            std::vector<double> prim = init_func(x);
            std::vector<double> cons(nvar);
            primitiveToConservative(prim, cons);
            U[i] = cons;
        }

    } else if constexpr (DIM == 2) {
        for (int i = 0; i < n[0]; i++) {
            for (int j = 0; j < n[1]; j++) {
                std::array<double, 2> x = {i * d[0], j * d[1]};
                std::vector<double> prim = init_func(x);
                std::vector<double> cons(nvar);
                primitiveToConservative(prim, cons);
                U[i][j] = cons;
            }
        }

    } else if constexpr (DIM == 3) {
        for (int i = 0; i < n[0]; i++) {
            for (int j = 0; j < n[1]; j++) {
                for (int k = 0; k < n[2]; k++) {
                    std::array<double, 3> x = {i * d[0], j * d[1], k * d[2]};
                    std::vector<double> prim = init_func(x);
                    std::vector<double> cons(nvar);
                    primitiveToConservative(prim, cons);
                    U[i][j][k] = cons;
                }
            }
        }
    }
}

// Advance solution by one time step using finite volume method
// dU/dt + dF/dx + dG/dy = 0  =>  U^(n+1) = U^n - dt/dx * (F_R - F_L) - dt/dy * (G_T - G_B)
template<int DIM>
void EulerSolverND<DIM>::timeStep(double dt) {
    // Copy current solution
    U_new = U;
    
    // Compute fluxes and update interior cells
    if constexpr (DIM == 1) {
        for (int i = 1; i < n[0]-1; i++) {
            std::vector<double> fluxL(nvar), fluxR(nvar);
            
            rusanovFlux(U[i-1], U[i], fluxL, 0);
            rusanovFlux(U[i], U[i+1], fluxR, 0);
            
            for (int k = 0; k < nvar; k++) {
                U_new[i][k] = U[i][k] - dt/d[0] * (fluxR[k] - fluxL[k]);
            }
        }

    } else if constexpr (DIM == 2) {
        for (int i = 1; i < n[0]-1; i++) {
            for (int j = 1; j < n[1]-1; j++) {
                std::vector<double> fluxL(nvar), fluxR(nvar), fluxB(nvar), fluxT(nvar);
                
                // X-direction
                rusanovFlux(U[i-1][j], U[i][j], fluxL, 0);
                rusanovFlux(U[i][j], U[i+1][j], fluxR, 0);
                
                // Y-direction
                rusanovFlux(U[i][j-1], U[i][j], fluxB, 1);
                rusanovFlux(U[i][j], U[i][j+1], fluxT, 1);
                
                for (int k = 0; k < nvar; k++) {
                    U_new[i][j][k] = U[i][j][k] - dt/d[0] * (fluxR[k] - fluxL[k]) - dt/d[1] * (fluxT[k] - fluxB[k]);
                }
            }
        }

    } else if constexpr (DIM == 3) {
        for (int i = 1; i < n[0]-1; i++) {
            for (int j = 1; j < n[1]-1; j++) {
                for (int k = 1; k < n[2]-1; k++) {
                    std::vector<double> fluxL(nvar), fluxR(nvar);
                    std::vector<double> fluxB(nvar), fluxT(nvar);
                    std::vector<double> fluxBk(nvar), fluxFr(nvar);
                    
                    // X-direction
                    rusanovFlux(U[i-1][j][k], U[i][j][k], fluxL, 0);
                    rusanovFlux(U[i][j][k], U[i+1][j][k], fluxR, 0);
                    
                    // Y-direction
                    rusanovFlux(U[i][j-1][k], U[i][j][k], fluxB, 1);
                    rusanovFlux(U[i][j][k], U[i][j+1][k], fluxT, 1);
                    
                    // Z-direction
                    rusanovFlux(U[i][j][k-1], U[i][j][k], fluxBk, 2);
                    rusanovFlux(U[i][j][k], U[i][j][k+1], fluxFr, 2);
                    
                    for (int k = 0; k < nvar; k++) {
                        U_new[i][j][k][k] = U[i][j][k][k] - dt/d[0] * (fluxR[k] - fluxL[k]) - dt/d[1] * (fluxT[k] - fluxB[k]) - dt/d[2] * (fluxFr[k] - fluxBk[k]);
                    }
                }
            }
        }
    }
    
    // Update solution
    U = U_new;
    
    // Apply boundary conditions
    applyBoundaryConditions();
}

// Write current solution to VTK file for visualization in ParaView
template<int DIM>
void EulerSolverND<DIM>::writeToVTK(const std::string& filename) const {
    std::ofstream file(filename);
    
    // VTK header
    file << "# vtk DataFile Version 3.0\n";
    file << DIM << "D Euler Solution\n";
    file << "ASCII\n";
    file << "DATASET STRUCTURED_GRID\n";
    
    // Grid dimensions
    if constexpr (DIM == 1) {
        file << "DIMENSIONS " << n[0] << " 1 1\n";
    } else if constexpr (DIM == 2) {
        file << "DIMENSIONS " << n[0] << " " << n[1] << " 1\n";
    } else if constexpr (DIM == 3) {
        file << "DIMENSIONS " << n[0] << " " << n[1] << " " << n[2] << "\n";
    }
    
    // Calculate total points
    int total_points = 1;
    for (int dim = 0; dim < DIM; dim++) {
        total_points *= n[dim];
    }
    
    // Points (grid coordinates)
    file << "POINTS " << total_points << " float\n";
    
    if constexpr (DIM == 1) {
        for (int i = 0; i < n[0]; i++) {
            file << i * d[0] << " 0.0 0.0\n";
        }

    } else if constexpr (DIM == 2) {
        for (int j = 0; j < n[1]; j++) {
            for (int i = 0; i < n[0]; i++) {
                file << i * d[0] << " " << j * d[1] << " 0.0\n";
            }
        }

    } else if constexpr (DIM == 3) {
        for (int k = 0; k < n[2]; k++) {
            for (int j = 0; j < n[1]; j++) {
                for (int i = 0; i < n[0]; i++) {
                    file << i * d[0] << " " << j * d[1] << " " << k * d[2] << "\n";
                }
            }
        }
    }
    
    // Point data (values at grid points)
    file << "POINT_DATA " << total_points << "\n";
    
    // Density
    file << "SCALARS density float 1\n";
    file << "LOOKUP_TABLE default\n";
    if constexpr (DIM == 1) {
        for (int i = 0; i < n[0]; i++) {
            std::vector<double> prim(nvar);
            conservativeToPrimitive(U[i], prim);
            file << prim[0] << "\n";
        }

    } else if constexpr (DIM == 2) {
        for (int j = 0; j < n[1]; j++) {
            for (int i = 0; i < n[0]; i++) {
                std::vector<double> prim(nvar);
                conservativeToPrimitive(U[i][j], prim);
                file << prim[0] << "\n";
            }
        }

    } else if constexpr (DIM == 3) {
        for (int k = 0; k < n[2]; k++) {
            for (int j = 0; j < n[1]; j++) {
                for (int i = 0; i < n[0]; i++) {
                    std::vector<double> prim(nvar);
                    conservativeToPrimitive(U[i][j][k], prim);
                    file << prim[0] << "\n";
                }
            }
        }
    }
    
    // Pressure
    file << "SCALARS pressure float 1\n";
    file << "LOOKUP_TABLE default\n";
    if constexpr (DIM == 1) {
        for (int i = 0; i < n[0]; i++) {
            std::vector<double> prim(nvar);
            conservativeToPrimitive(U[i], prim);
            file << prim[DIM + 1] << "\n";
        }

    } else if constexpr (DIM == 2) {
        for (int j = 0; j < n[1]; j++) {
            for (int i = 0; i < n[0]; i++) {
                std::vector<double> prim(nvar);
                conservativeToPrimitive(U[i][j], prim);
                file << prim[DIM + 1] << "\n";
            }
        }

    } else if constexpr (DIM == 3) {
        for (int k = 0; k < n[2]; k++) {
            for (int j = 0; j < n[1]; j++) {
                for (int i = 0; i < n[0]; i++) {
                    std::vector<double> prim(nvar);
                    conservativeToPrimitive(U[i][j][k], prim);
                    file << prim[DIM + 1] << "\n";
                }
            }
        }
    }
    
    // Velocity magnitude
    file << "SCALARS velocity_magnitude float 1\n";
    file << "LOOKUP_TABLE default\n";
    if constexpr (DIM == 1) {
        for (int i = 0; i < n[0]; i++) {
            std::vector<double> prim(nvar);
            conservativeToPrimitive(U[i], prim);
            double vel_mag = std::abs(prim[1]);
            file << vel_mag << "\n";
        }

    } else if constexpr (DIM == 2) {
        for (int j = 0; j < n[1]; j++) {
            for (int i = 0; i < n[0]; i++) {
                std::vector<double> prim(nvar);
                conservativeToPrimitive(U[i][j], prim);
                double vel_mag = sqrt(prim[1]*prim[1] + prim[2]*prim[2]);
                file << vel_mag << "\n";
            }
        }

    } else if constexpr (DIM == 3) {
        for (int k = 0; k < n[2]; k++) {
            for (int j = 0; j < n[1]; j++) {
                for (int i = 0; i < n[0]; i++) {
                    std::vector<double> prim(nvar);
                    conservativeToPrimitive(U[i][j][k], prim);
                    double vel_mag = sqrt(prim[1]*prim[1] + prim[2]*prim[2] + prim[3]*prim[3]);
                    file << vel_mag << "\n";
                }
            }
        }
    }
    
    // Velocity vector
    file << "VECTORS velocity float\n";
    if constexpr (DIM == 1) {
        for (int i = 0; i < n[0]; i++) {
            std::vector<double> prim(nvar);
            conservativeToPrimitive(U[i], prim);
            file << prim[1] << " 0.0 0.0\n";
        }

    } else if constexpr (DIM == 2) {
        for (int j = 0; j < n[1]; j++) {
            for (int i = 0; i < n[0]; i++) {
                std::vector<double> prim(nvar);
                conservativeToPrimitive(U[i][j], prim);
                file << prim[1] << " " << prim[2] << " 0.0\n";
            }
        }

    } else if constexpr (DIM == 3) {
        for (int k = 0; k < n[2]; k++) {
            for (int j = 0; j < n[1]; j++) {
                for (int i = 0; i < n[0]; i++) {
                    std::vector<double> prim(nvar);
                    conservativeToPrimitive(U[i][j][k], prim);
                    file << prim[1] << " " << prim[2] << " " << prim[3] << "\n";
                }
            }
        }
    }
    
    file.close();
}

// Main solver loop: evolve solution from t=0 to tmax
template<int DIM>
void EulerSolverND<DIM>::solve(double tmax, int output_interval, const std::string& output_prefix) {
    double t = 0.0;
    int step = 0;
    int file_counter = 0;
    
    std::cout << "Starting " << DIM << "D Euler simulation...\n";
    std::cout << "Grid: ";
    for (int dim = 0; dim < DIM; dim++) {
        std::cout << n[dim];
        if (dim < DIM - 1) std::cout << " x ";
    }
    std::cout << "\n";
    
    // Create output directory
    #ifdef _WIN32
        system("mkdir output 2>nul");
        std::string mkdir_cmd = "mkdir output\\" + output_prefix + " 2>nul";
        system(mkdir_cmd.c_str());
    #else
        system("mkdir -p ../output");
        std::string mkdir_cmd = "mkdir -p ../output/" + output_prefix;
        system(mkdir_cmd.c_str());
    #endif

    // Write initial condition
    char filename[200];
    sprintf(filename, "../output/%s/%s_%06d.vtk", output_prefix.c_str(), output_prefix.c_str(), file_counter);
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
            sprintf(filename, "../output/%s/%s_%06d.vtk", output_prefix.c_str(), output_prefix.c_str(), file_counter);
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
    sprintf(filename, "../output/%s/%s_%06d.vtk", output_prefix.c_str(), output_prefix.c_str(), file_counter);
    writeToVTK(filename);
    
    std::cout << "Simulation completed.\n";
    std::cout << "Output files in: ../output/" << output_prefix << "/\n";
}

// Explicit instantiations for 1D, 2D, and 3D
template class EulerSolverND<1>;
template class EulerSolverND<2>;
template class EulerSolverND<3>;