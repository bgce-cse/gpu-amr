
struct Euler <: Equation end
@declare_dofs Euler [:rhou, :rhov, :rho, :rhoE]

struct ShockTube <: Scenario end

function is_periodic_boundary(eq::Euler, scenario::GaussianWave)
    true
end

function is_periodic_boundary(eq::Euler, scenario::ShockTube)
    false
end

function is_analytical_solution(equation::Euler, scenario::GaussianWave)
    false
end

function is_analytical_solution(equation::Euler, scenario::ShockTube)
    false
end

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

function evaluate_pressure(eq::Euler, rho, E, imx, imy)
    0.4 .* (E .- 0.5 .* (imx.^2 .+ imy.^2) ./ rho)
end

function evaluate_flux(eq::Euler, celldofs, cellflux)
    x_coord = 1:size(celldofs, 1)
    ycoord = x_coord .+ size(celldofs, 1)
    
        p = evaluate_pressure(eq,
                          celldofs[:, 3],  # rho
                          celldofs[:, 4],  # E
                          celldofs[:, 1],  # rhou
                          celldofs[:, 2])  # rhov

    cellflux[x_coord, 1] .= celldofs[:,1] .* celldofs[:,1] ./ celldofs[:,3] .+ p
    cellflux[x_coord, 2] .= celldofs[:,1] .* celldofs[:,2] ./ celldofs[:,3]
    cellflux[x_coord, 3] .= celldofs[:,1]
    cellflux[x_coord, 4] .= celldofs[:,1] .* (celldofs[:,4] .+ p) ./ celldofs[:,3]
    
    cellflux[ycoord, 1] .= celldofs[:,1] .* celldofs[:,2] ./ celldofs[:,3]
    cellflux[ycoord, 2] .= celldofs[:,2] .* celldofs[:,2] ./ celldofs[:,3] .+ p
    cellflux[ycoord, 3] .= celldofs[:,2]
    cellflux[ycoord, 4] .= celldofs[:,2] .* (celldofs[:, 4] .+ p) ./ celldofs[:,3]
end

function max_eigenval(eq::Euler, celldata, normalidx; gamma=1.4)
    u = celldata[:, 1] ./ celldata[:, 3]
    v = celldata[:, 2] ./ celldata[:, 3]

    # compute pressure using evaluate_pressure, passing vectors directly
    p = evaluate_pressure(eq, celldata[:, 3], celldata[:, 4], celldata[:, 1], celldata[:, 2])

    c = sqrt.(gamma .* p ./ celldata[:, 3])

    vn = (normalidx == 1) ? u : v

    return maximum(abs.(vn) .+ c)
end

function evaluate_boundary(eq::Euler, scenario::ShockTube, face, normalidx, dofsface, dofsfaceneigh)
    # dofsface and dofsfaceneigh have shape (num_2d_quadpoints, dofs)
    dofsfaceneigh .= dofsface
    dofsfaceneigh[:, normalidx] .= -dofsface[:, normalidx]
end

