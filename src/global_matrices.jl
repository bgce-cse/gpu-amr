struct GlobalMatrices
    normalsigns::Dict{Face, Int64}
    normalidxs::Dict{Face, Int64}
    oppositefaces::Dict{Face, Face}
    project_dofs_to_face::Dict{Face,Array{Float64,2}}
    project_flux_to_face::Dict{Face,Array{Float64,2}}
    project_dofs_from_face::Dict{Face,Array{Float64,2}}

    quadweights_nd::Array{Float64, 1}
    quadpoints_nd::Array{Tuple{Float64, Float64}, 1}
    reference_massmatrix_face::Array{Float64,2}
    reference_massmatrix_cell::Array{Float64,2}
    reference_derivative_matrix::Array{Float64,2}

    filter_matrix::Array{Float64,2}

    function GlobalMatrices(basis::Basis, filter::Filter, dimensions)
        faces = instances(TerraDG.Face)

        normalsigns = Dict(W => -1, E => 1, N => 1, S => -1)
        normalidxs = Dict(W => 1, E => 1, N => 2, S => 2)
        oppositefaces = Dict(W => E, E => W, N => S, S => N)

        project_dofs_to_face = Dict(f => face_projection_matrix(basis, f) for f in faces)
        project_flux_to_face = Dict(f => kron(I(2), face_projection_matrix(basis, f)) for f in faces)
        project_dofs_from_face = Dict(f => face_projection_matrix(basis, f)' for f in faces)

        quadweights_nd = kron(basis.quadweights, basis.quadweights)


        nq = length(basis.quadpoints)
        quadpoints_nd = Vector{Tuple{Float64, Float64}}(undef, nq*nq)
        k = 1
        for j in 1:nq 
            for i in 1:nq 
                quadpoints_nd[k] = (basis.quadpoints[i], basis.quadpoints[j])
                k += 1
            end
        end

        reference_massmatrix_cell = massmatrix(basis, dimensions)
        reference_massmatrix_face = massmatrix(basis, dimensions - 1)
        reference_derivative_matrix = derivativematrix(basis)
        filter_matrix = evaluate_filter_matrix(filter, basis)

        new(normalsigns, normalidxs, oppositefaces,
            project_dofs_to_face, project_flux_to_face, project_dofs_from_face,
            quadweights_nd,
            quadpoints_nd,
            reference_massmatrix_face, reference_massmatrix_cell,
            reference_derivative_matrix, filter_matrix)
    end
end
