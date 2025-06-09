struct Acoustic <: Equation end
@declare_dofs Acoustic [:u, :v, :pressure, :rho, :K]
struct GaussianWave <: Scenario end

function get_initial_values(eq::Acoustic, scenario::GaussianWave, global_position; t=0.0)
    t = t
    x, y = global_position
    
    u = 0.0
    v = 0.0
    pressure = exp(-100*(x-0.5)^2 -100*(y-0.5)^2)
    rho = 1.0
    K = x <= 0.5 ? 0.2 : 1.0

    return [u, v, pressure, rho, K]
end

function is_analytical_solution(equation::Acoustic, scenario::GaussianWave)
    true
end

function evaluate_flux(eq::Acoustic, celldofs, cellflux)
    # Extract variables
    u = celldofs[:,1]  # x-velocity
    v = celldofs[:,2]  # y-velocity
    pressure = celldofs[:,3]  # pressure
    rho = celldofs[:,4]  # density
    K = celldofs[:,5]  # bulk modulus
    

    momentum_flux_x = pressure ./ rho  # Flux for u-momentum equation
    momentum_flux_y = pressure ./ rho  # Flux for v-momentum equation
    

    pressure_flux_x = K .* u  # Flux for pressure equation in x-direction
    pressure_flux_y = K .* v  # Flux for pressure equation in y-direction
    
    # Zero entries for variables that don't have flux in certain directions
    zero_entries = zeros(length(u))

    fx = [momentum_flux_x zero_entries pressure_flux_x zero_entries zero_entries]
    fy = [zero_entries momentum_flux_y pressure_flux_y zero_entries zero_entries]
    
    # Stack x and y fluxes
    cellflux .= vcat(fx, fy)
end

function max_eigenval(eq::Acoustic, celldata, normalidx)
    rho_vals = celldata[:, 4]
    K_vals = celldata[:, 5]
    rho_min = minimum(rho_vals)
    K_max = maximum(K_vals)
    return sqrt(K_max / rho_min)
end


# function evaluate_boundary(eq::Acoustic, scenario::GaussianWave, face,
# normalidx, dofsface, dofsfaceneigh)
# # dofsface and dofsfaceneigh have shape (num_2d_quadpoints, dofs)
# # you need to set dofsfaceneigh
# dofsfaceneigh .= dofsface
# dofsfaceneigh[:, normalidx] .= -dofsface[:, normalidx]


# end
"""
is_periodic_boundary(eq::Acoustic, scenario::GaussianWave)
The GaussianWave scenario does not require periodic boundary conditions.
"""
function is_periodic_boundary(eq::Acoustic, scenario::GaussianWave)
true
end





