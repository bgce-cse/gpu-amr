"""
    struct GlobalMatrices

Stores global matrices and some other things.
Everything that is very expensive to compute which stays
constant throughout the simulation should be stored here.


    GlobalMatrices(basis::Basis, filter::Filter, dimensions)

Initialize GlobalMatrices for `basis`, `filter` and `dimensions`.
"""
 struct GlobalMatrices
     normalsigns::Dict{Face, Int64}
     normalidxs::Dict{Face, Int64}
     oppositefaces::Dict{Face, Face}
     project_dofs_to_face::Dict{Face,Array{Float64,2}}
     project_flux_to_face::Dict{Face,Array{Float64,2}}
     project_dofs_from_face::Dict{Face,Array{Float64,2}}

     quadweights_nd::Array{Float64, 1}
     reference_massmatrix_face::Array{Float64,2}
     reference_massmatrix_cell::Array{Float64,2}
     reference_derivative_matrix::Array{Float64,2}

     filter_matrix::Array{Float64,2}

     function GlobalMatrices(basis::Basis, filter::Filter, dimensions)
        faces = instances(TerraDG.Face)

        normalsigns = Dict(left => -1, right => 1, top => 1, bottom => -1)
        normalidxs = Dict(left => 1, right => 1, top => 2, bottom => 2)
        oppositefaces = Dict(left => right, right => left, top => bottom, bottom => top)

        # TODO: Implement
        project_dofs_to_face = Dict(f => face_projection_matrix(basis, f) for f in faces)

        project_flux_to_face = Dict(f => kron(I(2),face_projection_matrix(basis, f)) for f in faces)
        project_dofs_from_face = Dict(f => face_projection_matrix(basis, f)' for f in faces)
        eigvals_mass = eigvals(massmatrix(basis, 2))
println("Mass matrix min eigenvalue: ", minimum(eigvals_mass))
println("Mass matrix max eigenvalue: ", maximum(eigvals_mass))
println("Mass matrix condition number: ", cond(massmatrix(basis, 2)))

        # project_flux_to_face = Dict(f => ones(1,2) for f in faces)
        # project_dofs_from_face = Dict(f => ones(1,1) for f in faces)

        # TODO: Implement
        basissize_1d = basis.order
        basissize_nd = basis.order ^ basis.dimensions
        quadweights_nd = kron(basis.quadweights,basis.quadweights)

        reference_massmatrix_cell = massmatrix(basis, dimensions)
        reference_massmatrix_face = massmatrix(basis, dimensions - 1)
        reference_derivative_matrix = derivativematrix(basis)

        filter_matrix = evaluate_filter_matrix(filter, basis)

         new(normalsigns, normalidxs, oppositefaces, project_dofs_to_face,
         project_flux_to_face, project_dofs_from_face, quadweights_nd, 
         reference_massmatrix_face,
         reference_massmatrix_cell, reference_derivative_matrix,
         filter_matrix)



     end
 end

