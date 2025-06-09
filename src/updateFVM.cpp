/*
 * updateFVM.cpp
 *
 * This file contains the updateFVM function. It computes updated velocity for a single time-step using FVM.
 *
 * Inputs:
 * - u, v: velocity
 * - p: pressure
 * - nx, ny: grid points
 * - dx, dy: grid spacing
 * - dt: time-step
 * - rho: density
 * - mu: dynamic viscosity
 *
 * Outputs:
 * - u_new, v_new: updated velocity
 * - p_new: updated pressure
 */

#include "updateFVM.hpp"
#include <vector>
#include <cmath>
#include <tuple>
#include <stdexcept>

std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>, std::vector<std::vector<double>>> updateFVM(
    std::vector<std::vector<double>>& u, // velocity in x
    std::vector<std::vector<double>>& v, // velocity in y
    std::vector<std::vector<double>>& p, // pressure
    int nx, int ny, // grid points
    double dx, double dy, // grid spacing
    double dt, // time-step
    double rho, // density
    double mu // dynamic viscosity
) {
    // grid structures for edges (to calculate cell averages)
    std::vector<std::vector<double>> flux_u_x(nx + 1, std::vector<double>(ny)); // flux of u across vertical edges
    std::vector<std::vector<double>> flux_u_y(nx, std::vector<double>(ny + 1)); // flux of u across horizontal edges
    std::vector<std::vector<double>> flux_v_x(nx + 1, std::vector<double>(ny)); // flux of v across vertical edges
    std::vector<std::vector<double>> flux_v_y(nx, std::vector<double>(ny + 1)); // flux of v across horizontal edges

    // copy values for new updates
    std::vector<std::vector<double>> u_new = u; // copy initial velocity in x
    std::vector<std::vector<double>> v_new = v; // copy initial velocity in y
    std::vector<std::vector<double>> p_new = p; // copy initial pressure

    // loop over vertical edges (between i and i+1)
    for (int i = 0; i <= nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            // convective fluxes of u momentum across vertical edges (averages)
            // can also use upwind schemes
            double u_edge = 0.0;
            double v_edge = 0.0;
            if (i == 0) { // boundary
                u_edge = u[0][j];
                v_edge = v[0][j];
            } else if (i == nx) { // boundary
                u_edge = u[nx - 1][j];
                v_edge = v[nx - 1][j];
            } else {
                u_edge = 0.5 * (u[i - 1][j] + u[i][j]);
                v_edge = 0.5 * (v[i - 1][j] + v[i][j]);
            }

            // convective flux
            // u momentum across vertical edge, flux = rho * u_edge * u_edge * area (dy)
            flux_u_x[i][j] = rho * u_edge * u_edge * dy;

            // v momentum across vertical edge, flux = rho * u_edge * v_edge * area (dy)
            flux_v_x[i][j] = rho * u_edge * v_edge * dy;

            // diffusive flux
            double du_dx = 0.0;
            double dv_dx = 0.0;
            if (i == 0) {
                du_dx = (u[i][j] - u[i][j]) / dx; // boundary, zero gradient
                dv_dx = (v[i][j] - v[i][j]) / dx;
            } else if (i == nx) {
                du_dx = (u[i - 1][j] - u[i - 1][j]) / dx; // boundary, zero gradient
                dv_dx = (v[i - 1][j] - v[i - 1][j]) / dx;
            } else {
                du_dx = (u[i][j] - u[i - 1][j]) / dx;
                dv_dx = (v[i][j] - v[i - 1][j]) / dx;
            }
            // flux = mu * (du/dx) * area (dy)
            flux_u_x[i][j] -= mu * du_dx * dy;
            flux_v_x[i][j] -= mu * dv_dx * dy;
        }
    }

    // loop over horizontal edges (between j and j+1)
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j <= ny; ++j) {
            // convective fluxes of u momentum across horizontal edges (averages)
            // can also use upwind schemes
            double u_edge = 0.0;
            double v_edge = 0.0;
            if (j == 0) { // boundary
                u_edge = u[i][0];
                v_edge = v[i][0];
            } else if (j == ny) { // boundary
                u_edge = u[i][ny - 1];
                v_edge = v[i][ny - 1];
            } else {
                u_edge = 0.5 * (u[i][j - 1] + u[i][j]);
                v_edge = 0.5 * (v[i][j - 1] + v[i][j]);
            }

            // convective flux for u and v momentum across horizontal edge
            // flux = rho * v_edge * u_edge * dx (for u momentum)
            flux_u_y[i][j] = rho * v_edge * u_edge * dx;

            // flux = rho * v_edge * v_edge * dx (for v momentum)
            flux_v_y[i][j] = rho * v_edge * v_edge * dx;

            // diffusive flux
            double du_dy = 0.0;
            double dv_dy = 0.0;
            if (j == 0) {
                du_dy = (u[i][j] - u[i][j]) / dy; // boundary, zero gradient
                dv_dy = (v[i][j] - v[i][j]) / dy;
            } else if (j == ny) {
                du_dy = (u[i][j - 1] - u[i][j - 1]) / dy; // boundary, zero gradient
                dv_dy = (v[i][j - 1] - v[i][j - 1]) / dy;
            } else {
                du_dy = (u[i][j] - u[i][j - 1]) / dy;
                dv_dy = (v[i][j] - v[i][j - 1]) / dy;
            }
            // flux = mu * (du/dx) * area (dy)
            flux_u_y[i][j] -= mu * du_dy * dx;
            flux_v_y[i][j] -= mu * dv_dy * dx;
        }
    }

    // update velocities
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            // net fluxes for u velocity in cell (i,j)
            double net_flux_u = flux_u_x[i][j] - flux_u_x[i + 1][j] + flux_u_y[i][j] - flux_u_y[i][j + 1];
            // net fluxes for v velocity in cell (i,j)
            double net_flux_v = flux_v_x[i][j] - flux_v_x[i + 1][j] + flux_v_y[i][j] - flux_v_y[i][j + 1];

            // update intermediate velocities using net fluxes
            u_new[i][j] = u[i][j] + dt * net_flux_u / (rho * dx * dy);
            v_new[i][j] = v[i][j] + dt * net_flux_v / (rho * dx * dy);
        }
    }
        
    // ppe for updated pressure values
    const int max_iter = 1000;
    const double tol = 1e-6;
    bool converged = false;

    for (int iter = 0; iter < max_iter; ++iter) {
        double residual = 0.0;

        for (int i = 1; i < nx - 1; ++i) {
            for (int j = 1; j < ny - 1; ++j) {
                double dudx = (u_new[i + 1][j] - u_new[i - 1][j]) / (2.0 * dx);
                double dvdy = (v_new[i][j + 1] - v_new[i][j - 1]) / (2.0 * dy);
                double rhs = (rho / dt) * (dudx + dvdy);
                double laplace_p = (
                    (p_new[i + 1][j] + p_new[i - 1][j]) * dy * dy +
                    (p_new[i][j + 1] + p_new[i][j - 1]) * dx * dx);

                double p_old = p_new[i][j];

                p_new[i][j] = (laplace_p - rhs * dx * dx * dy * dy) / (2.0 * (dx * dx + dy * dy));

                residual = std::max(residual, std::abs(p_new[i][j] - p_old));
            }
        }

        // apply pressure BC (Neumann)
        for (int i = 0; i < nx; ++i) {
            p_new[i][0] = p_new[i][1];
            p_new[i][ny - 1] = p_new[i][ny - 2];
        }
        for (int j = 0; j < ny; ++j) {
            p_new[0][j] = p_new[1][j];
            p_new[nx - 1][j] = p_new[nx - 2][j];
        }

        // exit loop if converged
        if (residual < tol) {
            converged = true;
            break;
        }
    }

    if (!converged) {
        throw std::runtime_error("Pressure Poisson solver did not converge: maximum iterations reached.");
    }

    p = p_new; // update p for correction step

    // pressure correction
    for (int i = 1; i < nx - 1; ++i) {
        for (int j = 1; j < ny - 1; ++j) {
            double dp_dx = (p[i + 1][j] - p[i - 1][j]) / (2 * dx);
            double dp_dy = (p[i][j + 1] - p[i][j - 1]) / (2 * dy);

            // update final velocity with pressure gradient
            u_new[i][j] -= dt * dp_dx / rho;
            v_new[i][j] -= dt * dp_dy / rho;
        }
    }

    // TODO: apply BC for velocities

    return std::make_tuple(u_new, v_new, p_new);
}
