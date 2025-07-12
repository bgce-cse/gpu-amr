using TerraDG
using Test
using LinearAlgebra

@testset "All tests" begin

@testset "Quadrature on the reference element" begin
    n = 5
    basis = TerraDG.Basis(n,1)
    p = basis.quadpoints
    wp = basis.quadweights
    func = (x,y) -> x*y*y
    sum = 0.0
    for i in CartesianIndices((n,n))
        sum += func(p[i[1]],p[i[2]]) * wp[i[1]]*wp[i[2]]
    end
    @test abs(sum - 1. / 6.) <= 10e-15
end

@testset "Maps reference coordinate to correct physical coord" begin
    cellcenter = [1.0, 1.0]
    cellsize = [0.5, 0.5]
    cell = TerraDG.Cell(cellcenter, 
            cellsize,
            [],
            [],
            -1)

    @test TerraDG.globalposition(cell, [0.5, 0.5]) == cellcenter
    @test TerraDG.globalposition(cell, [0.0, 0.0]) == [0.75, 0.75]
    @test TerraDG.globalposition(cell, [1.0, 1.0]) == [1.25, 1.25]
    @test_throws BoundsError TerraDG.globalposition(cell, [1.1, 0.0])
    @test_throws BoundsError TerraDG.globalposition(cell, [-0.1, 0.0])
end

@testset "Global to reference to global is same" begin
    cellcenter = [1.0, 1.0]
    cellsize = [0.5, 0.5]
    cell = TerraDG.Cell(cellcenter, 
            cellsize,
            [],
            [],
            -1)

    coords = [
        [0.5, 0.5],
        [0.0, 0.0],
        [1.0, 1.0]
    ]
    for coord ∈ coords
        global_coord = TerraDG.globalposition(cell, coord)
        local_coord = TerraDG.localposition(cell, global_coord)
        @test local_coord == coord
    end
end

@testset "Volume of 2D cells is correct" begin
    sizes = [
        [1.0, 1.0],
        [2.0, 2.0],
        [0.5, 0.5]
    ] 

    volumes = [
        1.0,
        4.0,
        0.25
    ]
    for (size, volume) ∈ zip(sizes, volumes)
        cell = TerraDG.Cell([0.0, 0.0], 
            size,
            [],
            [],
            -1)
        @test TerraDG.volume(cell) == volume
    end
end

@testset "Area of 2D cells is correct" begin
    sizes = [
        [1.0, 1.0],
        [2.0, 2.0],
        [0.5, 0.5]
    ] 

    areas = [
        1.0,
        2.0,
        0.5
    ]
    for (size, area) ∈ zip(sizes, areas)
        cell = TerraDG.Cell([0.0, 0.0], 
            size,
            [],
            [],
            -1)
        @test TerraDG.area(cell) == area
    end
end

@testset "Lagrange polynomials are correct" begin
    δ(a,b) = if (a == b) 1.0 else 0.0 end
    for n in range(1, length=6)
        basis = TerraDG.Basis(n, 1)
        points = basis.quadpoints
        for (i,p) in enumerate(points)
            for (j,q) in enumerate(points)
                @test TerraDG.lagrange_1d(points, i, q) == δ(i,j)
            end
        end
    end
end

@testset "Lagrange polynomials sum to one" begin
    for n in range(1, length=6)
        for x in [0.0, 0.5, 1.0]
            basis = TerraDG.Basis(n, 1)
            points = basis.quadpoints
            sum = 0.0
            for (i,p) in enumerate(points)
                sum += TerraDG.lagrange_1d(points, i, x)
            end
            @test isapprox(sum, 1, atol=10^-14)
        end
    end
end

@testset "Integral of derivatives of 1D-Lagrange polynomials is correct" begin
    Lvec = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    order = 10
    basis = TerraDG.Basis(order,1)
    for j in eachindex(Lvec)
        L = Lvec[j]
        for i in 1:length(basis.quadpoints)
            lhs = TerraDG.lagrange_1d(basis.quadpoints, i, L) - TerraDG.lagrange_1d(basis.quadpoints, i, 0)
            rhs = sum(L*basis.quadweights .* [TerraDG.lagrange_diff(basis.quadpoints, i, L*basis.quadpoints[n]) for n in 1:length(basis.quadpoints)])
            @test isapprox(lhs, rhs, atol=1e-14)
        end
    end
end

@testset "Integral of derivatives of 2D-Lagrange polynomials is correct" begin
    Lvec = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    order = 10
    basis = TerraDG.Basis(order,2)
    for L in Lvec
        for i in 1:length(basis.quadpoints)
            for j in 1:length(basis.quadpoints)

                lhs = sum(basis.quadweights .* [
                    (TerraDG.lagrange_1d(basis.quadpoints, i, L) * TerraDG.lagrange_1d(basis.quadpoints, j, basis.quadpoints[n])) - 
                    (TerraDG.lagrange_1d(basis.quadpoints, i, 0) * TerraDG.lagrange_1d(basis.quadpoints, j, basis.quadpoints[n]))
                    for n in 1:length(basis.quadpoints)
                ])

                rhs = sum([
                    (L * basis.quadweights[n1]) * basis.quadweights[n2] * 
                    TerraDG.lagrange_diff(basis.quadpoints, i, L * basis.quadpoints[n1]) * 
                    TerraDG.lagrange_1d(basis.quadpoints, j, basis.quadpoints[n2])
                    for n1 in 1:length(basis.quadpoints), n2 in 1:length(basis.quadpoints)
                ])

                @test isapprox(lhs, rhs, atol=1e-14)
            end
        end
    end
end

@testset "Derivative of const = 0" begin
    for n=1:6
        basis = TerraDG.Basis(n, 1)
        for x in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            @test isapprox(
            sum([TerraDG.lagrange_diff(basis.quadpoints, i, x) for i=1:size(basis, 1)]), 0, atol=10e-14)
        end
    end
end

@testset "Mass matrix is correct for 2D" begin
    basis_order1 = TerraDG.Basis(1, 2)
    basis_order2 = TerraDG.Basis(2, 2)
    @test TerraDG.massmatrix(basis_order1, 2) == reshape([1], 1, 1)
    massmatrix_order2 = Diagonal([0.25,0.25,0.25,0.25])
    @test TerraDG.massmatrix(basis_order2, 2) == massmatrix_order2
end

@testset "Projection/evaluation works for polynomials" begin
    ns = [1,2,3,4,5,6]
    funs = [
        (x, y) -> [1],
        (x, y) -> [x + y],
        (x, y) -> [(x + y)^2],
        (x, y) -> [(x + y)^3],
        (x, y) -> [(x + y)^4],
        (x, y) -> [(x + y)^5]
    ]
    points = [
        [0.21, 0.23],
        [0.38, 0.93],
        [0.92, 0.23],
        [0.01, 0.01],
        [0.99, 0.99],
    ]
    proj_reshaped(basis, func) = reshape(
        TerraDG.project_to_reference_basis(func, basis, 1), length(basis.quadweights)^2 )

    for (n, fun) in zip(ns, funs)
        basis = TerraDG.Basis(n, 2)
        coeffs = proj_reshaped(basis, fun)
        for point in points
            evaluated = TerraDG.evaluate_basis(basis, coeffs, point)
            reference = fun(point[1], point[2])[1]
            @test isapprox(evaluated, reference, atol=10e-14)
        end
    end
end

@testset "Derivative matrix is correct" begin
    ns = [1,2,3,2,3,4,5,6,2,5]
    funs = [
        (x, y) -> [1]
        (x, y) -> [1]
        (x, y) -> [1]
        (x, y) -> [x + y]
        (x, y) -> [(x + y)^2]
        (x, y) -> [(x + y)^3]
        (x, y) -> [(x + y)^4]
        (x, y) -> [(x + y)^5]
        (x,y) ->  [x*y]
        (x, y) -> [(x^3 + y^4 + x^2 * y^2)]
    ]
    funs_deriv_x = [
        (x,y) -> [0]
        (x,y) -> [0]
        (x,y) -> [0]
        (x,y) -> [1]
        (x,y) -> [2 * (x + y)]
        (x,y) -> [3 * (x + y)^2]
        (x,y) -> [4 * (x + y)^3]
        (x,y) -> [5 * (x + y)^4]
        (x,y) -> [y]
        (x,y) -> [x * (3*x + 2*y^2)]
    ]
    funs_deriv_y = [
        (x,y) -> [0]
        (x,y) -> [0]
        (x,y) -> [0]
        (x,y) -> [1]
        (x,y) -> [2 * (x + y)]
        (x,y) -> [3 * (x + y)^2]
        (x,y) -> [4 * (x + y)^3]
        (x,y) -> [5 * (x + y)^4]
        (x,y) -> [x]
        (x,y) -> [2 * y * (x^2 + 2*y^2)]
    ]

    for (n, fun, deriv_x, deriv_y) in zip(ns, funs, funs_deriv_x, funs_deriv_y)
        proj_reshaped(basis, func) = reshape(
            TerraDG.project_to_reference_basis(func, basis, 1), length(basis.quadweights)^2 )
        basis = TerraDG.Basis(n, 2)
        fun_evaluated = proj_reshaped(basis, fun)
        reference_derivx_evaluated = proj_reshaped(basis, deriv_x)
        reference_derivy_evaluated = proj_reshaped(basis, deriv_y)
        ∇ = TerraDG.derivativematrix(basis)
        derivx_evaluated = transpose(∇[:, 1:length(basis)]) * fun_evaluated
        derivy_evaluated = transpose(∇[:, length(basis) + 1 : end]) * fun_evaluated

        @test all(isapprox.(derivx_evaluated, reference_derivx_evaluated, atol=10e-12))
        @test all(isapprox.(derivy_evaluated, reference_derivy_evaluated, atol=10e-12))
    end
end

@testset "Face projection matrix is correct" begin
    ns = [1,2,3,4,5,6]
    funs = [
        (x, y) -> [1,1,1]
        (x, y) -> [0,0,0]
        (x, y) -> [0,1,0]
    ]
    for n in ns
        basis = TerraDG.Basis(n, 2)
        for func in funs
            func_proj = TerraDG.project_to_reference_basis(func, basis, 3)
            for face in [TerraDG.N, TerraDG.S, TerraDG.W, TerraDG.E]
                fp = TerraDG.face_projection_matrix(basis, TerraDG.top)
                func_proj_face = fp * func_proj
                @test sum(func_proj)/length(func_proj) ≈ sum(func_proj_face)/length(func_proj_face)
            end
        end
    end
end

end