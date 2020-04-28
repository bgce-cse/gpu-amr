"""
    evaluate_error(eq::Equation, scenario::Scenario, grid::Grid, t::Float64)

Prints out the L1/L2/L∞ integral norms of the error for
equation `eq`, scenario (with analytical solution) `scenario`, grid
`grid` at time `t`.
Error is defined as difference between analytical solution and numerical
approximation.
"""
function evaluate_error(eq::Equation, scenario::Scenario, grid::Grid, t::Float64)
    @assert is_analytical_solution(eq, scenario)
    l1_error = similar(grid.dofs, get_ndofs(eq))
    l2_error = similar(l1_error)
    linf_error = similar(l1_error)
    l1_error .= 0
    l2_error .= 0
    linf_error .= 0
    basis = grid.basis
    basissize_1d = size(basis,1)
    lin_idx = LinearIndices((
        basissize_1d, basissize_1d
    ))
    for cell in grid.cells
        for i=1:basissize_1d
            for j=1:basissize_1d
                x = basis.quadpoints[i]
                y = basis.quadpoints[j]
                quadweight = basis.quadweights[i] * basis.quadweights[j]
                weight = volume(cell) * quadweight
                pxg, pyg = globalposition(cell, (x,y))
                analytical = get_initial_values(eq, scenario, (pxg,pyg), t=t)
                for var=1:get_ndofs(eq)
                    coeff_idx = lin_idx[i,j]
                    value = grid.dofs[coeff_idx, var, cell.dataidx]

                    error = abs(value - analytical[var])
                    l1_error[var] += weight * error
                    l2_error[var] += weight * error^2
                    linf_error[var] = max(linf_error[var], error)
                end
            end
        end
    end

    l2_error .= sqrt.(l2_error)
    println("Errors for each variable.")
    println("Var\tL1\tL2\tL∞")
    for i=1:get_ndofs(eq)
        var_name = string(get_variable_name(eq,i))
        @printf "%s\t%15.6e\t%15.6e\t%15.6e\n" var_name l1_error[i] l2_error[i] linf_error[i]
    end

end