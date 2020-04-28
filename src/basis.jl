using FastGaussQuadrature

"""
    lagrange_1d(points, i, x)

Evaluate the Lagrange interpolation polynomial defined over nodal `points` with index `i`
at point `x`.
"""
function lagrange_1d(points, i, x)
    1.0
end

"""
    lagrange_diff(points, i, x)

Evaluate the derivative of the Lagrange interpolation polynomial defined over nodal `points` 
with index `i` at point `x`.
"""
function lagrange_diff(points, i, x)
    0.0
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
Base.length(basis::Basis) = 1

"""
    Base.size(basis::Basis)

Return number of points for `basis` for each dimensions as tuple.
"""
function Base.size(basis::Basis) 
    ntuple(basis.dimensions) do x 
       1
    end
end

"""
    Base.size(basis::Basis, dim)

Return number of points for `basis` for dimensions `dim`.
"""
function Base.size(basis::Basis, dim::Integer)
    1
end

"""
    evaluate_basis(basis::Basis, coeffs, x)

Evaluate the `basis` with coefficients
`coeffs` at point `x`.
"""
function evaluate_basis(basis::Basis, coeffs, x)
    coeffs[1]
end

"""
    project_to_reference_basis(fun, basis::Basis, ndofs)

Project the result of the function `fun` to coefficients
of the basis built of a tensor-product of `basis`.
The function `fun(x,y)`  takes in the ``x, y``-coordinates
and returns a vector with size `ndofs`.
The corresponding coefficients are returned.
"""
function project_to_reference_basis(fun, basis::Basis, ndofs)
    reshape(fun(0.5, 0.5), (1,ndofs))
end


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
    ones(1.0, 1.0)
end

"""
    face_projection_matrix(basis, face)

Return the face projection matrix for `basis` and `face`.
Multiplying it with coefficient vector for the right basis
returns the coefficients of the solution evaluated at the 
quadrature nodes of the face.
"""
function face_projection_matrix(basis, face)
    ones(1,1)
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
