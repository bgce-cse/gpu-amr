
struct Limiter
    center_avg::Array{Float64, 1}
    neigh_avg::Dict{Face, Array{Float64, 1}}
    center_s::Array{Float64, 2}
    neigh_s::Array{Float64, 2}

    function Limiter(ndofs)
        faces = instances(TerraDG.Face)

        center_avg = zeros(ndofs)
        neigh_avg = Dict(f => similar(center_avg) for f in faces)

        center_s = zeros(2, ndofs)
        neigh_s = zeros(4, ndofs)
        new(center_avg, neigh_avg, center_s, neigh_s)
    end
end

function make_limiter(config::Configuration, ndofs)
    if config.limiter_name == "none" || isnothing(config.limiter_name)
        return nothing
    elseif config.limiter_name == "minmod"
        return Limiter(ndofs)
    else
        error(string("Unknown limiter name: ", config.limiter_name))
    end
end

function evaluate_integral(globals, proj, dofs; f = x -> 1, c = 1.0)
    total = zero(proj)
    for i in eachindex(globals.quadweights_nd)
        weight = globals.quadweights_nd[i]
        point  = globals.quadpoints_nd[i]
        total += weight * f(point) * dofs[i, :]
    end
    proj .= c * total
end

function minmod(s1, s2, s3)
    if sign(s1) == sign(s2) == sign(s3)
        sign(s1) * min(abs(s1), abs(s2), abs(s3))
    else
        0
    end 
end

function linearize(globals, center_avg, slope_x, slope_y, dofs, cell)
    for i = eachindex(globals.quadpoints_nd)
        x, y = globalposition(cell, globals.quadpoints_nd[i])
        dofs[i] = center_avg + (x - cell.center[1]) * slope_x + (y - cell.center[2]) * slope_y
    end
end

function limit_slopes(globals, limiter, dofs, cell)

    # compute own slopes by projection
    @views evaluate_integral(globals, limiter.center_s[1,:], dofs; f = ((x,_),) -> x - 0.5, c=12)
    @views evaluate_integral(globals, limiter.center_s[2,:], dofs; f = ((_,y),) -> y - 0.5, c=12)

    # compute neighbor slopes
    dx_inv = 1 / cell.size[1]
    dy_inv = 1 / cell.size[2]

    limiter.center_s[1,:] .*= dx_inv
    limiter.center_s[2,:] .*= dy_inv
    
    # compute neighbor averages
    limiter.neigh_s[1,:] .= (limiter.center_avg .- limiter.neigh_avg[left]) .* dx_inv
    limiter.neigh_s[2,:] .= (limiter.neigh_avg[right] .- limiter.center_avg) .* dx_inv

    limiter.neigh_s[3,:] .= (limiter.center_avg .- limiter.neigh_avg[bottom]) .* dy_inv
    limiter.neigh_s[4,:] .= (limiter.neigh_avg[top] .- limiter.center_avg) .* dy_inv

    for i = 1:size(dofs, 2)
        slope_x = minmod(limiter.center_s[1, i], limiter.neigh_s[1, i], limiter.neigh_s[2, i])
        slope_y = minmod(limiter.center_s[2, i], limiter.neigh_s[3, i], limiter.neigh_s[4, i])

        if slope_x != limiter.center_s[1, i] || slope_y != limiter.center_s[2, i]
            @views linearize(globals, limiter.center_avg[i], slope_x, slope_y, dofs[:,i], cell)            
        end
    end
end

function limit(grid, globals, ::Nothing)
    # No limiting required
    return
end

function limit(grid, globals, limiter::Limiter)
    faces = [left, top, right, bottom]
    for i in eachindex(grid.cells)
        @views cell = grid.cells[i]
        @views data = grid.dofs[:,:, cell.dataidx]

        evaluate_integral(globals, limiter.center_avg, data)
        for (i, neigh) in enumerate(cell.neighbors)

            if cell.facetypes[i] == boundary
                limiter.neigh_avg[faces[i]] .= 0
            else
                evaluate_integral(globals, limiter.neigh_avg[faces[i]], grid.dofs[:,:,neigh.dataidx])
            end
        end

        limit_slopes(globals, limiter, data, cell)
    end
end