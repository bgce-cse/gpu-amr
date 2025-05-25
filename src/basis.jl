using FastGaussQuadrature

"""
    lagrange_1d(points, i, x)

Evaluate the Lagrange interpolation polynomial defined over nodal `points` with index `i`
at point `x`.
"""
function lagrange_1d(points, i, x)
    return prod(x .- points[1:i-1])*prod(x .- points[i+1:end]) / (prod(points[i] .- points[1:i-1])*prod(points[i] .- points[i+1:end]))
end

"""
    lagrange_diff(points, i, x)

Evaluate the derivative of the Lagrange interpolation polynomial defined over nodal `points` 
with index `i` at point `x`.
"""
function lagrange_diff(points, i, x)
    return lagrange_1d(points, i, x) * (sum(1.0./(x .- points[1:i-1])) + sum(1.0./(x .- points[i+1:end])) )
    
end

"""
    get_quadpoints(n)

Compute quadrature points and weights for Gaussian quadrature
of order `n`.
The points (and thus the corresponding weights) are normalized to the range
``[0.0, 1.0]``.

Return a tuple of `points, weights`.
"""
function get_quadpoints(n)
    x, w = gausslegendre(n)
    # They are scaled in [-1.0, 1.0]
    # and we need [0.0, 1.0]
    (x .+ 1)./2, w ./ 2
end

"""
    Basis

A standard 1-dimensional basis of `order::Integer`
with `quadpoints::Array{Float64,1}` and 
corresponding `quadweights::Array{Float64,1}`
Is the basis (pun intended) for tensor-product
bases.

    Basis(order::Integer, dimensions)

Initialize a basis of `order::Integer` and
dimensions `dimensions`.
"""
struct Basis
    quadpoints::Array{Float64,1}
    quadweights::Array{Float64,1}
    order::Int64
    dimensions::Int64

    function Basis(order, dimensions)
        quadpoints, quadweights = get_quadpoints(order)
        new(quadpoints, quadweights, order, dimensions)
    end
end


"""
    Base.length(basis::Basis)

Return number of points for `basis` in n-dimensions.
"""
Base.length(basis::Basis) = length(basis.quadpoints)^basis.dimensions

"""
    Base.size(basis::Basis)

Return number of points for `basis` for each dimensions as tuple.
"""
function Base.size(basis::Basis) 
    ntuple(_->length(basis.quadpoints),basis.dimensions)
end

"""
    Base.size(basis::Basis, dim)

Return number of points for `basis` for dimensions `dim`.
"""
function Base.size(basis::Basis, dim::Integer)
    @boundscheck 1 <= dim <= basis.dimensions || throw(BoundsError(basis.dimensions, dim))
    size(basis)[dim]
end

"""
    evaluate_basis(basis::Basis, coeffs, x)

Evaluate the `basis` with coefficients
`coeffs` at point `x`.
"""
function evaluate_basis(basis::Basis, coeffs, x)

    C = reshape(coeffs,size(basis,1),size(basis,2))
    sum([C[i,j]*lagrange_1d(basis.quadpoints,i,x[1])*lagrange_1d(basis.quadpoints,j,x[2]) for i in 1:size(basis,1) for j in 1:size(basis,2)])
    
   
end


"""
    project_to_reference_basis(fun, basis::Basis, ndofs)

Project the result of the function `fun` to coefficients
of the basis built of a tensor-product of `basis`.
The function `fun(x,y)`  takes in the ``x, y``-coordinates
and returns a vector with size `ndofs`.
The corresponding coefficients are returned.
"""
function project_to_reference_basis(fun, basis::Basis, ndofs::Integer)
    n   = basis.order
    pts = basis.quadpoints

    M = zeros(n*n, ndofs)

    # Loop once over every (i,j) pair in natural column-major order:
    for (row, idx) in enumerate(CartesianIndices((n, n)))
        i, j          = Tuple(idx)       # unpack the CartesianIndex
        x, y          = pts[i], pts[j]   # reference coords
        M[row, :]     = fun(x, y)        # fill that row
    end

    return M
end#return order*order *3


"""
    massmatrix(basis, dimensions)

Return the mass-matrix for a `dimensions`-dimensional
tensor-product basis built up from the 1d-basis `basis`.
"""
function massmatrix(basis, dimensions)
    ones(1,1)
end

"""
    derivativematrix(basis)

Returns the 2-dimensional derivative matrix for `basis`.
Multiplying this with flux-coefficients of shape
`(dimensions * basissize_2d, ndofs)` returns the
coefficients of the corresponding derivative.
"""
function derivativematrix(basis)
    zeros(1, basis.dimensions * 1)
end


"""
    get_face_quadpoints(basis::Basis, face)

Return the quadrature points at the face `face` for basis `basis`.
"""
function get_face_quadpoints(basis::Basis, face)
    if face == left
        # x = 0, y varies
        return (0.0, basis.quadpoints)
    elseif face == right
        # x = 1, y varies
        return (1.0, p)
    elseif face == bottom
        # y = 0, x varies
        return (basis.quadpoints, 0.0)
    elseif face == top
        # y = 1, x varies
        return (basis.quadpoints, 1.0)
    else
        error("Unknown face")
    end
end

"""
    face_projection_matrix(basis, face)

Return the face projection matrix for `basis` and `face`.
Multiplying it with coefficient vector for the right basis
returns the coefficients of the solution evaluated at the 
quadrature nodes of the face.
"""
# function face_projection_matrix(basis, face)
#     n = length(basis.quadpoints)
#     n2d = n^2
    
#     # Create face projection matrix
#     P = zeros(n, n2d)
    
#     # Get quadrature points on the face
#     face_points = get_face_quadpoints(basis, face)


    
#     # Fill the projection matrix
#     for i in 1:n  # Row index (face point index)
#         x, y = face_points[i]
        
#         for j in 1:n  # Column indices (volume basis functions)
#             for k in 1:n
#                 idx = (k-1)*n + j  # Linear index for 2D basis
                
#                 # Evaluate basis function at face point
#                 if face == left || face == right
#                     # For left/right faces, y varies along the face
#                     P[i, idx] = lagrange_1d(basis.quadpoints, j, x) * 
#                                 lagrange_1d(basis.quadpoints, k, y)
#                 else  # top or bottom
#                     # For top/bottom faces, x varies along the face
#                     P[i, idx] = lagrange_1d(basis.quadpoints, j, x) * 
#                                 lagrange_1d(basis.quadpoints, k, y)
#                 end
#             end
#         end
#     end
    
#     return P
# end

function face_projection_matrix(basis, face)
    face_quad = get_face_quadpoints(basis,face)
    if face == left || face == right
        phi = [lagrange_1d(face_quad[2],j,face_quad[1]) for j in 1:length(basis.quadpoints)]
    else 
        phi = [lagrange_1d(face_quad[1],j,face_quad[2]) for j in 1:length(basis.quadpoints)]
    end
    print(phi)
    LinearAlgebra.kron(LinearAlgebra.I(basis.order),phi)'

end


"""
    evaluate_m_to_n_vandermonde_basis(basis) 

Return the Vandermonde matrix that converts between the 2D-modal 
(normalized) Legendre-basis and the 2D-nodal tensor-product basis built
with `basis`.
"""
function evaluate_m_to_n_vandermonde_basis(basis)
    Array{Float64}(I, length(basis), length(basis))
end
