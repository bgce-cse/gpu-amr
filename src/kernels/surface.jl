"""
    struct BuffersFaceIntegral

Stores all temporary buffers needed during face integrals.
Avoids costly re-allocation.
"""
struct BuffersFaceIntegral
    dofsface::Array{Float64,2}
    dofsfaceneigh::Array{Float64,2}
    fluxface::Array{Float64,2}
    fluxfaceneigh::Array{Float64,2}
    numericalflux::Array{Float64,2}
    numericalflux_scaled::Array{Float64,2}

    function BuffersFaceIntegral(basis, ndofs)
        @assert basis.dimensions == 2
        basissize_1d = size(basis, 1)
        basissize_nd = length(basis)
        dofsface = zeros(basissize_1d,ndofs)
        dofsfaceneigh = similar(dofsface)
        fluxface = zeros(basissize_1d * 2,ndofs)
        fluxfaceneigh = similar(fluxface)
        numericalflux = similar(dofsface)
        numericalflux_scaled = similar(numericalflux)

        new(dofsface, dofsfaceneigh, fluxface, fluxfaceneigh, numericalflux, numericalflux_scaled)
    end
end

"""
    rusanov(eq, dofs, dofsneigh, flux, fluxneigh, dx, normalidx, normalsign, numericalflux)

Computes the Rusanov (or local Lax-Friedrichs) numerical flux for degrees of freedom
`dofs`, degrees of freedoms of the neighbor `dofsneigh`, flux `flux`, flux of neighbor
`fluxneigh`, cellsize `dx`.
All quantities are assumed to be represented by a basis on the reference line.
The face is parametrized by a index `normalidx`, where 1 stands for a face in x-direction
and 2 for a face in y-direction.
The sign of the outer normal of the face is given by `normalsign`.

The numerical flux is stored in `numericalflux`.
Method also returns the maximal eigenvalue.
"""
function rusanov(eq, dofs, dofsneigh, flux, fluxneigh, dx, normalidx, normalsign, numericalflux)
    maxeigenval_center = max_eigenval(eq, dofs, normalidx)
    maxeigenval_neigh = max_eigenval(eq, dofsneigh, normalidx)
    maxeigenval = max(maxeigenval_center, maxeigenval_neigh)
    
    #where does normalisgn go (first/second term)?
    #where does dx go?

    first_term = 0.5 .* normalsign .* (flux[normalidx, :] .+ fluxneigh[normalidx, :])
    second_term = 0.5 .* maxeigenval .* (dofs .- dofsneigh)
    
    numericalflux .= dx .* (first_term' .+ second_term)
 
    return maxeigenval
end


"""
    project_to_faces(globals, dofs, flux, dofsface, fluxface, face)

Projects degrees of freedom `dofs` and `flux` to `face`.
Result is stored in `dofsface` and `fluxface`.
Projection matrices are stored in `globals`.
"""
function project_to_faces(globals, dofs, flux, dofsface, fluxface, face)
    
    dofsface .= globals.project_dofs_to_face[face] * dofs
    fluxface .= globals.project_flux_to_face[face] * flux# TODO fix this 
   
end

"""
    evaluate_face_integral(eq, globals, buffers, cell, face, celldu)

Computes face integrals for equation `eq`, cell `cell` on face `face`.
Global matrices are passed in `globals`, buffers in `buffers`.
Result is stored in `celldu`.
"""
function evaluate_face_integral(eq, globals, buffers, cell, face, celldu)
    normalidx = globals.normalidxs[face]
    normalsign = globals.normalsigns[face]

    # Compute Riemann solver (in normal)
    buffers.numericalflux .= 0
    maxeigenval = rusanov(eq, buffers.dofsface, buffers.dofsfaceneigh, buffers.fluxface, buffers.fluxfaceneigh, cell.size[1], normalidx, normalsign, buffers.numericalflux)

    # TODO Modify celldu with update from face integral
    celldu .-= buffers.numericalflux

    return maxeigenval

end
