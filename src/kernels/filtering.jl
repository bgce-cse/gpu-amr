"""
    abstract type Filter

Abstract type for filter.
"""
abstract type Filter end

"""
    requires_filtering(filter::Filter)

Return `true` if `filter` is non-trivial, i.e.,
is not the identity operation.
"""
function requires_filtering(filter::Filter)
    true
end

"""
    struct IdentityFilter <: Filter

Trivial filter. Does nothing.
"""
struct IdentityFilter <: Filter end

"""
    evaluate_filter_scaling_matrix(filter::IdentityFilter, basis)

Return the scaling matrix for our `IdentityFIlter`.
Is the identity matrix with correct shape.
"""
function evaluate_filter_scaling_matrix(filter::IdentityFilter, basis)
    @assert basis.dimensions == 2
    basissize_2d = length(basis)
    Matrix{Float64}(I, basissize_2d, basissize_2d)
end

"""
    requires_filtering(filter::IdentityFilter)

As this filter does nothing, we do not need to apply it.
"""
function requires_filtering(filter::IdentityFilter)
    false
end

"""
    evaluate_filter_matrix(filter::Filter, basis)

Return the filter matrix for `filter` and `basis`.
You can apply this to a vector of degrees of freedom to surpress 
high-frequency parts.
"""
function evaluate_filter_matrix(filter::Filter, basis)
    basissize_2d = length(basis)
    Matrix{Float64}(I, basissize_2d, basissize_2d)
end

"""
    make_filter(config::Configuration)

Returns filter for configuration `config`.
"""
function make_filter(config::Configuration)
    if config.filter_name == "identity"
        filter = IdentityFilter()
    else
        error(string("Unknown filter name: ", config.filter_name))
    end
    filter
end
