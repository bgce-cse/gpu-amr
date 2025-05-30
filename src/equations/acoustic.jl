struct Acoustic <: Equation end
@declare_dofs Acoustic [:u, :v, :pressure, :rho, :K]
struct GaussianWave <: Scenario end




function is_periodic_boundary(equation::Acoustic, scenario::GaussianWave)
    true
end

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

    velocities_flux = celldofs[:,3]./celldofs[:,4]
    pressure_flux_x = celldofs[:,5].*celldofs[:,1]
    pressure_flux_y = celldofs[:,5].*celldofs[:,2]
    zero_entries = zeros(length(velocities_flux))

    fx = [velocities_flux zero_entries pressure_flux_x zero_entries zero_entries]
    fy = [zero_entries velocities_flux pressure_flux_y zero_entries zero_entries]
    # stack them so you get 
    cellflux .= vcat(fx, fy)
end

function max_eigenval(eq::Acoustic, celldata, normalidx)
    rho = minimum(celldata[:, 4])######à######
    K = maximum(celldata[:,5])
    abs(sqrt(abs(K/rho)))
end
























