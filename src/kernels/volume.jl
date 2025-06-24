"""
    BuffersVolume(basis, ndofs)

This gives buffers that are used to store intermediate
results during `evaluate_volume`.
"""
struct BuffersVolume
    scaled_fluxcoeff::Array{Float64,2}
    chi::Diagonal{Float64, Array{Float64, 1}} 

    function BuffersVolume(basis, ndofs)
        basissize_nd = length(basis)

        scaled_fluxcoeff = zeros(basissize_nd * 2,ndofs)
        scaling = Diagonal(zeros(basissize_nd * 2, basissize_nd * 2))

        new(scaled_fluxcoeff, scaling)
    end
end

"""
    evaluate_volume(globals, buffers, flux_coeff, basis, inverse_jacobian, volume, celldu)

Evaluates the volume term of our pde.

# Arguments
- `globals`: struct containing global matrices
- `buffers`: buffers to store intermediate results
- `flux_coeff`: coefficients of the flux
- `basis`: 1d-basis
- `inverse_jacobian`: inverse of the Jacobian matrix. Assumed to be diagonal.
- `volume`: n-dimensional volume of the cell. E.g., area of 2d-cell
- `celldu`: update of dofs, return value is added to this
"""
function evaluate_volume(globals, buffers, flux_coeff, basis, inverse_jacobian, volume, celldu)
    quadweights = globals.quadweights_nd

    A = inverse_jacobian' * volume 

    buffers.chi .= kron(Diagonal(diag(A)), Diagonal(vec(quadweights)))

    celldu .+= globals.reference_derivative_matrix * (diag(buffers.chi) .* flux_coeff) #TODO sum factorization possible
end


