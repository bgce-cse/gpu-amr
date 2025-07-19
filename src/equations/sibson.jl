struct Sibson <: Equation end
@declare_dofs Sibson [:f]

struct ConcentricWaves <: Scenario
end

function is_periodic_boundary(equation::Sibson, scenario::ConcentricWaves)
    true
end

function get_initial_values(eq::Sibson, scenario::ConcentricWaves, global_position; t=0.0)
    t = t
    x, y = global_position
    f = cos(4*pi*sqrt((x-0.25)^2 + (y-0.25)^2))       
    return [f]
end

function is_analytical_solution(equation::Sibson, scenario::ConcentricWaves)
    true
end

function evaluate_flux(eq::Sibson, celldofs, cellflux)
    # velocity = 1.0
    # cellflux .= vcat([velocity .* celldofs for _ in 1:size(cellflux, 1)]...)
end

function max_eigenval(eq::Sibson, celldata, normalidx)
    1.0
end
