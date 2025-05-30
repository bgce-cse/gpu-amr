"This abstract type needs to be implemented by all equations."
abstract type Equation end

"This abstract type needs to be implementred by all scenarios"
abstract type Scenario end

"""
This function initializes the discrete solution at t = 0 for an equation `eq` and a scenario `scenario`.
It evaluates get\\_initial\\_values for `eq` and `scenario` at the global coordinates of our nodes.
It then interpolates this on the `basis`.
"""
function interpolate_initial_dofs(eq::Equation, scenario::Scenario, celldofs, cell, basis)
    celldofs .= project_to_reference_basis(basis, get_ndofs(eq)) do px, py
        pxg, pyg = globalposition(cell, (px, py)) #this fetch the coordinate from the global domain based on the local 
        get_initial_values(eq, scenario, (pxg, pyg))
    end
end

"This function returns the number of variables for equation `eq::Equation`"
function get_nvars(eq::Equation)
    get_ndofs(eq) - get_nparams(eq)
end

"This function returns the number of material parameters for equation `eq::Equation`."
function get_nparams(eq::Equation)
    0
end

"""

    get_variable_name(eq::Equation, variable_index)

Return the name of the variable with index `variable_index` for equation `eq::Equation`.
# Implementation
Custom equations need to define `get_variable_names(eq)` for this to work.
"""
function get_variable_name(eq::Equation, variable_index)
    get_variable_names(eq)[variable_index]
end

"""
    is_periodic_boundary(equation::Equation, scenario::Scenario) 

Returns true if all boundary conditions should be periodic for equation `eq::Equation` and
scenario `scenario::Scenario`.
"""
function is_periodic_boundary(equation::Equation, scenario::Scenario)
    false
end

"""
    evaluate_boundary(eq::Equation, scenario::Equation, face, normalidx, dofsface, dofsfaceneigh)

Evaluate the boundary condition, defined for equation `eq` and `scenario`, on face `face` with `normalidx`.
Store dofs of ghost cell in `dofsfaceneigh`.
You can use inner (face)-dofs `dofsface` to construct to boundary.
"""
function evaluate_boundary(eq::Equation, scenario::Equation, face, normalidx, dofsface, dofsfaceneigh)
    error("evaluate_boundary called but not define.")
end



"""
    get_initial_values(eq::Equation, scenario::Scenario, global_position; t=0.0) 

Return a vector of initial values for equation `eq` and scenario `scenario` at 
`global_positon` and at time `t`.
Values other than `t = 0` only have to be supported if
`is_analytical_solution` returns true.
"""
function get_initial_values(eq::Equation, scenario::Scenario, global_position; t=0.0) 
    error("Function get_initial_values not defined for equation/scenario")
end

"""
    is_analytical_solution(equation::Equation, scenario::Scenario)

Returns true if the initial condition is also an analytical solution 
for equation `equation` and scenario `scenario`.
"""
function is_analytical_solution(equation::Equation, scenario::Scenario)
    false
end

"""
    evaluate_flux(eq::Equation, celldofs, cellflux)

Evaluate the flux for equation `eq` and dofs `celldofs`,
Stores them in `cellflux`.
"""
function evaluate_flux(eq::Equation, celldofs, cellflux)
    error("Function evaluate_flux not defined for equation/scenario")
end

"""
    max_eigenval(eq::Equation, celldata, normalidx)

Return the maximum eigenvalue in direction of `normalidx` for
`celldata`.

"""
function max_eigenval(eq::Equation, celldata, normalidx)
    error("Function max_eigenval not defined for equation/scenario")
end



"""
    declare_dofs(eq, dof_names, num_params=0)

Generate boilerplate code for equation type `eq` which has
degrees of freedom named `dof_names` (should be iterable of symbols)
and `num_params` parameters.
Generates

    get_variable_names(eq)
    get_ndofs(eq)
    get_nvars(eq)
    get_nparams(eq)

Also generates a struct called [typename]Shortcuts.

# Example

    struct Advection <: Equation end
    @declare_dofs Advection [:ρ_1, :ρ_2, :ρ_3]

Generates:

    get_variable_names(eq::Advection) = [:ρ_1, :ρ_2, :ρ_3]
    get_ndofs(eq::Advection) = 3
    get_nvars(eq::Advection) = 3
    get_nparams(eq::Advection) = 0

You can use the struct in this way:

    s = AdvectionShortcuts()
    @assert s.ρ_1 == 1
    @assert s.ρ_2 == 2
    @assert s.ρ_3 == 3
"""
macro declare_dofs(eq, dof_names, num_params=0)
    local expressions::Array{Expr, 1} = []
    local shortcut_struct = Symbol(eq, "Shortcuts")
    push!(expressions, quote
        function $(esc(Symbol("get_variable_names")))(eq::$eq)
            $dof_names
        end

        function $(esc(Symbol("get_ndofs")))(eq::$eq)
            $(length(dof_names.args))
        end

        function $(esc(Symbol("get_nvars")))(eq::$eq)
            $(length(dof_names.args) - num_params)
        end

        function $(esc(Symbol("get_nparams")))(eq::$eq)
            $num_params
        end

        struct $(esc(shortcut_struct))
            
        end

        @inline function Base.getproperty(x::$(esc(shortcut_struct)), name::Symbol)
            get_variable_shortcut(x, Val(name))
        end
    end
    )

    for (i, name) in enumerate(dof_names.args)
        push!(expressions, quote
            @inline function $(esc(Symbol("get_variable_shortcut")))(eq::$(esc(shortcut_struct)), ::Val{$name})
                return $i
            end

            @inline function $(esc(Symbol("get_variable_shortcut")))(eq::$eq, ::Val{$name})
                return $i
            end
        end)
        end


    quote $(expressions...) end
end

#TODO is there a way to include all files in equations/?
include("equations/advection.jl")
include("equations/sibson.jl")

"""
    make_equation(config::Configuration)

Returns equation object for configuration `config`.
"""
function make_equation(config::Configuration)
    #TODO create list instead of a bunch of conditions
    if config.equation_name == "advection"
        return Advection()
    elseif config.equation_name =="sibson"
        return Sibson()
    else
        error(string("Unknown equation name: ", config.equation_name))
    end
end

"""
    make_scenario(config::Configuration)

Returns scenario object for configuration `config`.
"""
function make_scenario(config::Configuration)
    #TODO create list instead of a bunch of conditions
    if config.scenario_name == "planar_waves"
        return PlanarWaves()
    elseif config.scenario_name == "concentric_waves"
        return ConcentricWaves()
    else
        error(string("Unknown scenario name: ", scenario_name))
    end
end
