struct Advection <: Equation end
@declare_dofs Advection [:ρ1, :ρ2, :ρ3]

struct PlanarWaves <: Scenario
end

function is_periodic_boundary(equation::Advection, scenario::PlanarWaves)
    true
end

function get_initial_values(eq::Advection, scenario::PlanarWaves, global_position; t=0.0)
    x, y = global_position
    ρ1 = sin(2π*(x + y - 2*t))
    ρ2 = sin(2π*(y - t))   
    ρ3 = 1.0                    
    return [ρ1, ρ2, ρ3]
end

function is_analytical_solution(equation::Advection, scenario::PlanarWaves)
    true
end

function evaluate_flux(eq::Advection, celldofs, cellflux)
    velocity = 1.0
    fx = velocity .* celldofs 
    fy = velocity .* celldofs

    cellflux .= vcat(fx, fy)
end

function max_eigenval(eq::Advection, celldata, normalidx)
    1.0
end