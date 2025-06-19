
struct BuffersLimiter
    average::Array{Float64, 1}
    averages_neigh::Dict{Face, Array{Float64, 1}}
    own_slopes::Array{Float64, 2}
    slopes_neigh::Array{Float64, 2}

    function BuffersLimiter(ndofs)
        faces = instances(TerraDG.Face)

        average = zeros(ndofs)
        averages_neigh = Dict(f => similar(average) for f in faces)

        own_slopes = zeros(2, ndofs)
        slopes_neigh = zeros(4, ndofs)
        new(average, averages_neigh, own_slopes, slopes_neigh)
    end
end

function project_to_basis(globals, proj, dofs; f = x -> 1, c=1.0)
    proj .= c * sum(i -> globals.quadweights_nd[i] * f(globals.quadpoints_nd[i]) * dofs[i,:], eachindex(globals.quadweights_nd))
end

function minmod(s₁, s₂, s₃)
    if sign(s₁) == sign(s₂) == sign(s₃)
        sign(s₁) * min(abs(s₁), abs(s₂), abs(s₃))
    else
        0
    end 
end

function linearize(globals, average, slope_x, slope_y, dofs, cell)
    for i = eachindex(globals.quadpoints_nd)
        x, y = globalposition(cell, globals.quadpoints_nd[i])
        dofs[i] = average + (x - cell.center[1]) * slope_x + (y - cell.center[2]) * slope_y
    end
end

function limit_slopes(globals, buffers, dofs, cell)

    # compute own slopes by projection
    @views project_to_basis(globals, buffers.own_slopes[1,:], dofs; f = ((x,_),) -> x - 0.5, c=12)
    @views project_to_basis(globals, buffers.own_slopes[2,:], dofs; f = ((_,y),) -> y - 0.5, c=12)

    # compute neighbor slopes
    dx_inv = 1 / cell.size[1]
    dy_inv = 1 / cell.size[2]
    #divide b and c by dx
    buffers.own_slopes[1,:] .*= dx_inv
    buffers.own_slopes[2,:] .*= dy_inv
    # x direction
    buffers.slopes_neigh[1,:] .= (buffers.average .- buffers.averages_neigh[left]) .* dx_inv
    buffers.slopes_neigh[2,:] .= (buffers.averages_neigh[right] .- buffers.average) .* dx_inv

    # y direction
    buffers.slopes_neigh[3,:] .= (buffers.average .- buffers.averages_neigh[bottom]) .* dy_inv
    buffers.slopes_neigh[4,:] .= (buffers.averages_neigh[top] .- buffers.average) .* dy_inv

    for i = 1:size(dofs, 2)
        slope_x = minmod(buffers.own_slopes[1, i], buffers.slopes_neigh[1, i], buffers.slopes_neigh[2, i])
        slope_y = minmod(buffers.own_slopes[2, i], buffers.slopes_neigh[3, i], buffers.slopes_neigh[4, i])

        if slope_x != buffers.own_slopes[1, i] || slope_y != buffers.own_slopes[2, i]
            @views linearize(globals, buffers.average[i], slope_x, slope_y, dofs[:,i], cell)            
        end
    end
end

function limit(grid, globals, buffers)
    faces = [left, top, right, bottom]
    for i in eachindex(grid.cells)
        @views cell = grid.cells[i]
        @views data = grid.dofs[:,:, cell.dataidx]

        project_to_basis(globals, buffers.average, data)
        for (i, neigh) in enumerate(cell.neighbors)
            # ensures slope = 0 at boundary
            if cell.facetypes[i] == boundary
                buffers.averages_neigh[faces[i]] .= buffers.average
            else
                project_to_basis(globals, buffers.averages_neigh[faces[i]], grid.dofs[:,:,neigh.dataidx])
            end
        end

        limit_slopes(globals, buffers, data, cell)
    end
end