struct Advection <: Equation end
@declare_dofs Advection [:ρ1, :ρ2, :ρ3]

struct PlanarWaves <: Scenario
end

function is_periodic_boundary(equation::Advection, scenario::PlanarWaves)
    true
end

function get_initial_values(eq::Advection, scenario::PlanarWaves, global_position; t=0.0)
    [1.0, 1.0, 1.0]
end

function is_analytical_solution(equation::Advection, scenario::PlanarWaves)
    true
end


function evaluate_flux(eq::Advection, celldofs, cellflux)
    s = AdvectionShortcuts()
    # you can use s.ρ_1 == 1 etc to simplify stuff
    cellflux .= 0.0
end

function max_eigenval(eq::Advection, celldata, normalidx)
    # Is actually correct!
    1.0
end