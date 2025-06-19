
struct Euler <: Equation end
@declare_dofs Euler [:rhou, :rhov, :rho, :rhoE]


struct ShockTube <: Scenario end

function evaluate_energy(eq::Euler, rhou, rhov, rho, p, gamma = 1.4)
    p/(gamma - 1) + 1/(2*rho)*(rhou^2+rhov^2)
end

function get_initial_values(eq::Euler, scenario::GaussianWave, global_position; t=0.0)
   pxg, pyg = global_position
    p = exp(-100 * (pxg - 0.5)^2 - 100 *(pyg - 0.5)^2) + 1
    rho = 1.0
    rhoE = evaluate_energy(eq, 0,0,rho,p)
    [0.0, 0.0, rho, rhoE]
end

function get_initial_values(eq::Euler, scenario::ShockTube, global_position; t=0.0)
    pxg, pyg = global_position

    if pxg < 0.5 || pyg < 0.5
        rho = 0.125
        p = 0.1
    else
        rho = 1.0
        p = 1.0
    end

    rhoE = evaluate_energy(eq, 0,0,rho,p)
    [0.0, 0.0, rho, rhoE]
end


function is_analytical_solution(equation::Euler, scenario::GaussianWave)
    true
end
function is_analytical_solution(equation::Euler, scenario::ShockTube)
    true
end

function evaluate_flux(eq::Euler, celldofs, cellflux; gamma=1.4)
    rhou = celldofs[:, 1]
    rhov = celldofs[:, 2]
    rho  = celldofs[:, 3]
    rhoE = celldofs[:, 4]

    u = rhou ./ rho
    v = rhov ./ rho

    kinetic_energy = 0.5 * (rhou.^2 + rhov.^2) ./ rho
    p = (gamma - 1) .* (rhoE .- kinetic_energy)

    fx1 = (rhou .* rhov) .* rho .+ p
    fx2 = (rhou .* rhov) .* rho
    fx3 = rhou
    fx4 = u .* (rhoE .+ p)

    fy1 = (rhou .* rhov) .* rho
    fy2 = (rhou .* rhov) .* rho .+ p
    fy3 = rhov
    fy4 = v .* (rhoE .+ p)

    fx = hcat(fx1, fx2, fx3, fx4)
    fy = hcat(fy1, fy2, fy3, fy4)

    cellflux .= vcat(fx, fy)
end


function max_eigenval(eq::Euler, celldata, normalidx; gamma=1.4)
    rhou = celldata[:, 1]
    rhov = celldata[:, 2]
    rho  = celldata[:, 3]
    rhoE = celldata[:, 4]

    u = rhou ./ rho
    v = rhov ./ rho

    kinetic_energy = 0.5 * (rhou.^2 + rhov.^2) ./ rho
    p = (gamma - 1) .* (rhoE .- kinetic_energy)

    c = sqrt.(gamma * p ./ rho)

    if normalidx == 1
        vn = u
    else
        vn = v
    end

    return maximum(abs.(vn) .+ c)
end



function evaluate_boundary(eq::Euler, scenario::ShockTube, face, normalidx, dofsface, dofsfaceneigh)
# dofsface and dofsfaceneigh have shape (num_2d_quadpoints, dofs)
# you need to set dofsfaceneigh
dofsfaceneigh .= dofsface
dofsfaceneigh[:, normalidx] .= -dofsface[:, normalidx]


end

function is_periodic_boundary(eq::Euler, scenario::GaussianWave)
true
end
function is_periodic_boundary(eq::Euler, scenario::ShockTube)
false
end



