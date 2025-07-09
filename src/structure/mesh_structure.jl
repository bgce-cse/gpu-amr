

abstract type AbstractMesh end

function create_struct(amr::Bool, config::Configuration, eq::Equation, scenario::Scenario)
    if amr
        return AMRQuadTree(config, eq, scenario)
    else
        return make_grid(config, eq, scenario)
    end
end

