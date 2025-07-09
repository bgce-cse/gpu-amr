"""
    abstract type TimeIntegrator

Abstract type for ODE-Integrators.
They should store the update as internal state to 
avoid costly reallocations.

# Implementation
For new integrators, the method `step` needs to be 
overwritten.
This method should modify the dofs stored in the grid
in-place.
"""
abstract type TimeIntegrator end

"""
    ExplicitEuler(grid::AbstractMesh)

Return Euler time-integrator for `grid`.
First order accurate.
"""
struct ExplicitEuler 
    dofsupdate::Array{Float64,3}

    function ExplicitEuler(grid::AbstractMesh)
        dofsupdate = similar(grid.dofs)
        new(dofsupdate)
    end
end

"""
    step(f, integrator::ExplicitEuler, grid, dt)

Performs an update with the explicit euler method
on `grid` and timestepsize `dt`.
"""
function step(f, integrator::ExplicitEuler, grid, dt)
    integrator.dofsupdate .= 0.0
    f(integrator.dofsupdate, grid.dofs, grid.time)
    grid.dofs .+= dt .* integrator.dofsupdate
    @info "Update has" norm(integrator.dofsupdate)
end

"""
    SSPRK2(grid::AbstractMesh)

Return SSPRK2 time-integrator for `grid`.
Two-stage strong-stability preserving Runge Kutta method.
Second order accurate.
"""
struct SSPRK2
    stage1::Array{Float64,3}
    stage2::Array{Float64,3}

    function SSPRK2(grid::AbstractMesh)
        stage1 = similar(grid.dofs)
        stage2 = similar(grid.dofs)
        new(stage1, stage2)

    end
end

"""
    step(f, integrator::SSPRK2, grid, dt)

Performs an update with the SSPRK2 method
on `grid` and timestepsize `dt`.
"""

function step(f, integrator::SSPRK2, grid, dt)
    integrator.stage1 .= 0.0
    integrator.stage2 .= 0.0

    f(integrator.stage1, grid.dofs, grid.time)
    integrator.stage1 .= grid.dofs .+ dt .* integrator.stage1

    f(integrator.stage2, integrator.stage1, grid.time + dt)

    grid.dofs .= 0.5 .* (grid.dofs .+ integrator.stage1 .+
        dt .* integrator.stage2)
end

"""
    SSPRK3(grid::AbstractMesh)

Return SSPRK3 time-integrator for `grid`.
Three-stage strong-stability preserving Runge Kutta method.
Third order accurate.
"""
struct SSPRK3
    stage1::Array{Float64,3}
    stage2::Array{Float64,3}
    stage3::Array{Float64,3}

    function SSPRK3(grid::AbstractMesh)
        stage1 = similar(grid.dofs)
        stage2 = similar(grid.dofs)
        stage3 = similar(grid.dofs)
        new(stage1, stage2, stage3)

    end
end

"""
    step(f, integrator::SSPRK3, grid, dt)

Performs an update with the SSPRK3 method
on `grid` and timestepsize `dt`.
"""
function step(f, integrator::SSPRK3, grid, dt)
    integrator.stage1 .= 0.0
    integrator.stage2 .= 0.0
    integrator.stage3 .= 0.0

    f(integrator.stage1, grid.dofs, grid.time)
    integrator.stage1 .= grid.dofs .+ dt .* integrator.stage1

    f(integrator.stage2, integrator.stage1, grid.time + dt)
    integrator.stage2 .= 0.75 .* grid.dofs .+ 0.25 .* (
        integrator.stage1 + dt *integrator.stage2)

    f(integrator.stage3, integrator.stage2, grid.time + 0.5 * dt)


    grid.dofs .= 1.0/3.0 .* grid.dofs .+ 2.0/3.0 * (
        integrator.stage2 .+ dt .* integrator.stage3)
end

"""
    make_timeintegrator(config::Configuration, grid::AbstractMesh)

Returns time integrator from `config` for `grid`.
"""
function make_timeintegrator(config::Configuration, grid::AbstractMesh)
    if config.timeintegrator_name == "Euler"
        return ExplicitEuler(grid)
    elseif config.timeintegrator_name == "SSPRK2"
        return SSPRK2(grid)
    elseif config.timeintegrator_name == "SSPRK3"
        return SSPRK3(grid)
    else
        error(string("Unknown timeintegrator name: ", config.timeintegrator_name))
    end
end