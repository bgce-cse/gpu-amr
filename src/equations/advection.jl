struct Advection <: Equation end
@declare_dofs Advection [:ρ1, :ρ2, :ρ3]

struct PlanarWaves <: Scenario
end

function is_periodic_boundary(equation::Advection, scenario::PlanarWaves)
    true
end

function get_initial_values(eq::Advection, scenario::PlanarWaves, global_position; t=0.0)
    return [sin(2*pi*(global_position[1]+global_position[2])), sin(2*pi*(global_position[2])), 1.0]
    #return [1.0, 1.0, 1.0]
end

function is_analytical_solution(equation::Advection, scenario::PlanarWaves)
    true
end

function evaluate_flux(eq::Advection, celldofs, cellflux)
    cellflux = [1.0* celldofs, 1.0* celldofs] #courant?
end

function max_eigenval(eq::Advection, celldata, normalidx)
    # Is actually correct!
    1.0
end

#ciaociaocioa