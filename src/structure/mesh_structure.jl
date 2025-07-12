
"""
    @enum FaceType regular=1 boundary=2

Regular faces are faces that have another cell as neighbor.
Boundary faces are faces on the boundary, i.e., where we need
to construct a solution at each timestep.
"""
@enum FaceType regular=1 boundary=2

"""
    @enum Face left=1 top=2 right=3 bottom=4

`Face` describes the ordering of our faces. The order is irrelevant
as long as the same order is used everywhere.
Using the wrong order leads to very hard bugs!
"""
@enum Face W=1 N=2 E=3 S=4

abstract type AbstractMesh end

function create_struct(amr::Bool, config::Configuration, eq::Equation, scenario::Scenario)
    if amr
        return AMRQuadTree(config, eq, scenario)
    else
        return make_grid(config, eq, scenario)
    end
end

