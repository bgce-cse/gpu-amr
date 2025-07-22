using FastGaussQuadrature

"""
    lagrange_1d(points, i, x)

Evaluate the Lagrange interpolation polynomial defined over nodal `points` with index `i`
at point `x`.
"""
function lagrange_1d(points, i, x)
    return prod(x .- points[collect(1:end) .!=i]) / prod(points[i] .- points[collect(1:end) .!=i])
end

"""
    lagrange_diff(points, j, x)

Evaluate the derivative of the Lagrange interpolation polynomial defined over nodal `points`
with index `j` at point `x`.
"""
function lagrange_diff(points, j, x)
    # using formula that avoids 1/(x-quadpoints) due to division by zero for L=1
    sum([
        (1 / (points[j] - points[i])) *
        prod((x .- points[setdiff(1:end, (i, j))]) ./ (points[j] .- points[setdiff(1:end, (i, j))]))
        for i in 1:length(points) if i != j
    ])
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
    n   = basis.order
    pts = basis.quadpoints
    
    phi = [lagrange_1d(pts, i, x[1]) for i in 1:n]
    psi = [lagrange_1d(pts, j, x[2]) for j in 1:n]
    
    s = 0.0
    for (idx, cart_idx) in enumerate(CartesianIndices((n, n)))
        i, j = Tuple(cart_idx)
        s += coeffs[idx] * phi[i] * psi[j]
    end
    
    return s
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

    for (row, idx) in enumerate(CartesianIndices((n, n)))
        i, j = Tuple(idx)       
        x, y = pts[i], pts[j]   
        M[row, :] = fun(x, y)        
    end

    return M
end


"""
    massmatrix(basis, dimensions)

Return the mass-matrix for a `dimensions`-dimensional
tensor-product basis built up from the 1d-basis `basis`.
"""
function massmatrix(basis::Basis, dimensions)

    M1 = Diagonal(basis.quadweights)

    M = M1
    for _ in 2:dimensions
        M = kron(M, M1)
    end
    return M
end

"""
    derivativematrix(basis)

Returns the 2-dimensional derivative matrix for `basis`.
Multiplying this with flux-coefficients of shape
`(dimensions * basissize_2d, ndofs)` returns the
coefficients of the corresponding derivative.
""" 

function derivativematrix(basis::Basis)
    pts = basis.quadpoints
    n = basis.order

    D1 = [ lagrange_diff(pts, j, pts[i]) 
           for i in 1:n, j in 1:n]

    Dx = kron(I(n), D1)' 
    Dy = kron(D1, I(n))' 
    
    hcat(Dx,Dy)
end

"""
    get_face_quadpoints(basis::Basis, face)

Return the quadrature points at the face `face` for basis `basis`.
"""
function get_face_quadpoints(basis::Basis, face)
    if face == W
        # x = 0, y varies
        return (0.0, basis.quadpoints)
    elseif face == E
        # x = 1, y varies
        return (1.0, basis.quadpoints)
    elseif face == S
        # y = 0, x varies
        return (basis.quadpoints, 0.0)
    elseif face == N
        # y = 1, x varies
        return (basis.quadpoints, 1.0)
    else
        error("Unknown face")
    end
end


function face_projection_matrix(basis::Basis, face)
    n = basis.order
    if face == W || face == E
        # For vertical faces (x=0 or x=1), project along x-dimension
        # The 1D basis functions are evaluated at the fixed x-coordinate of the face
        x_face_coord = get_face_quadpoints(basis, face)[1]
        phi_x_at_face = [lagrange_1d(basis.quadpoints, i, x_face_coord) for i in 1:n]
        return kron(I(n), phi_x_at_face')
    else 
        # For horizontal faces (y=0 or y=1), project along y-dimension
        # The 1D basis functions are evaluated at the fixed y-coordinate of the face
        y_face_coord = get_face_quadpoints(basis, face)[2]
        phi_y_at_face = [lagrange_1d(basis.quadpoints, j, y_face_coord) for j in 1:n]
        return kron(phi_y_at_face', I(n))
    end
end
"""
Handle projection when neighbor is smaller than current cell.
"""
function project_smaller_neighbor!(
    basis, cell, sub_neigh, dofsneigh, fluxneigh, 
    buffers_face, faceneigh,face_dir
)
    face_points = get_face_quadpoints(basis, faceneigh)
    
    if Face(face_dir) == N || Face(face_dir) == S
        # North/South faces
        mask = (cell.center[1] < sub_neigh.center[1]) ? (face_points[1] .>= 0.5) : (face_points[1] .< 0.5)
        offset = (cell.center[1] < sub_neigh.center[1]) ? -1.0 : 0.0
        masked_indices = findall(mask)
        points = (face_points[1] .* 2 .+ offset, face_points[2])

        for (masked_idx, real_idx) in enumerate(masked_indices)
            buffers_face.dofsfaceneigh[real_idx, :] = evaluate_basis(basis, dofsneigh, [points[1][real_idx] points[2]])
        end

        for (masked_idx, real_idx) in enumerate(masked_indices)
            buffers_face.fluxfaceneigh[real_idx, :] = evaluate_basis(basis, fluxneigh[1:size(fluxneigh,1)÷2, :], [points[1][real_idx], points[2]])
            buffers_face.fluxfaceneigh[real_idx + basis.order, :] = evaluate_basis(basis, fluxneigh[size(fluxneigh,1)÷2+1:end, :], [points[1][real_idx], points[2]])
        end
    else
        # East/West faces
        mask = (cell.center[2] < sub_neigh.center[2]) ? (face_points[2] .>= 0.5) : (face_points[2] .< 0.5)
        offset = (cell.center[2] < sub_neigh.center[2]) ? -1.0 : 0.0
        masked_indices = findall(mask)
        points = (face_points[1], face_points[2] .* 2 .+ offset)

        for (masked_idx, real_idx) in enumerate(masked_indices)
            buffers_face.dofsfaceneigh[real_idx, :] = evaluate_basis(basis, dofsneigh, [points[1], points[2][real_idx]])
        end

        for (masked_idx, real_idx) in enumerate(masked_indices)
            buffers_face.fluxfaceneigh[real_idx, :] = evaluate_basis(basis, fluxneigh[1:size(fluxneigh,1)÷2, :], [points[1], points[2][real_idx]])
            buffers_face.fluxfaceneigh[real_idx + basis.order, :] = evaluate_basis(basis, fluxneigh[size(fluxneigh,1)÷2+1:end, :], [points[1], points[2][real_idx]])
        end
    end
end

"""
Handle projection when neighbor is larger than current cell.
"""
function project_larger_neighbor!(
    basis, cell, sub_neigh, dofsneigh, fluxneigh, 
    buffers_face, faceneigh, face_dir
)
    face_points = get_face_quadpoints(basis, faceneigh)

    if Face(face_dir) == N || Face(face_dir) == S
        offset = (cell.center[1] < sub_neigh.center[1]) ? 0.0 : 0.5
        points = (face_points[1] ./ 2 .+ offset, face_points[2])
        
        for idx in 1:basis.order
            buffers_face.dofsfaceneigh[idx, :] = evaluate_basis(basis, dofsneigh, [points[1][idx], points[2]])
        end
        
        for idx in 1:basis.order
            buffers_face.fluxfaceneigh[idx, :] = evaluate_basis(basis, fluxneigh[1:size(fluxneigh,1)÷2, :], [points[1][idx], points[2]])
            buffers_face.fluxfaceneigh[idx + basis.order, :] = evaluate_basis(basis, fluxneigh[size(fluxneigh,1)÷2+1:end, :], [points[1][idx], points[2]])
        end
    else
        # East/West faces
        offset = (cell.center[2] < sub_neigh.center[2]) ? 0.0 : 0.5
        points = (face_points[1], face_points[2] ./ 2 .+ offset)
        
        for idx in 1:basis.order
            buffers_face.dofsfaceneigh[idx, :] = evaluate_basis(basis, dofsneigh, [points[1], points[2][idx]])
        end
        
        for idx in 1:basis.order
            buffers_face.fluxfaceneigh[idx, :] = evaluate_basis(basis, fluxneigh[1:size(fluxneigh,1)÷2, :], [points[1], points[2][idx]])
            buffers_face.fluxfaceneigh[idx + basis.order, :] = evaluate_basis(basis, fluxneigh[size(fluxneigh,1)÷2+1:end, :], [points[1], points[2][idx]])
        end
    end
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
