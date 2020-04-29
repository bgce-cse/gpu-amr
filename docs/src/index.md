# TerraDG.jl Documentation

```@contents
```

# Main
```@docs
TerraDG.evaluate_rhs
TerraDG.main
```
# I/O
## Configuration
```@docs
TerraDG.Configuration
```

## VTK/Paraview output
```@docs
TerraDG.VTKPlotter
TerraDG.plot
TerraDG.save
```

## Error Writer
```@docs
TerraDG.evaluate_error
```

# Equations
```@docs
TerraDG.Equation
TerraDG.make_equation
TerraDG.Scenario
TerraDG.make_scenario
TerraDG.interpolate_initial_dofs
TerraDG.get_nvars
TerraDG.get_nparams
TerraDG.get_variable_name
TerraDG.is_periodic_boundary
TerraDG.evaluate_boundary
TerraDG.get_initial_values
TerraDG.is_analytical_solution
TerraDG.evaluate_flux
TerraDG.max_eigenval
TerraDG.@declare_dofs
```

# Grid
```@docs
TerraDG.FaceType
TerraDG.Cell
TerraDG.Grid
TerraDG.Face
TerraDG.get_neighbor
TerraDG.make_mesh
TerraDG.make_grid
TerraDG.globalposition
TerraDG.localposition
TerraDG.volume
TerraDG.area
TerraDG.inverse_jacobian
```

## Basis
```@docs
TerraDG.lagrange_1d
TerraDG.lagrange_diff
TerraDG.get_quadpoints
TerraDG.Basis
Base.length(::TerraDG.Basis)
Base.size(::TerraDG.Basis)
Base.size(::TerraDG.Basis,::Integer)
TerraDG.evaluate_basis
TerraDG.project_to_reference_basis
TerraDG.massmatrix
TerraDG.derivativematrix
TerraDG.get_face_quadpoints
TerraDG.face_projection_matrix
TerraDG.evaluate_m_to_n_vandermonde_basis
```

# Kernels
Many kernels take both global matrices and buffers.
Try to use them to avoid costly re-computations or memory
allocations.

## Global Matrices
```@docs
TerraDG.GlobalMatrices
```

## Time
```@docs
TerraDG.TimeIntegrator
TerraDG.make_timeintegrator
TerraDG.step
TerraDG.ExplicitEuler
TerraDG.SSPRK2
TerraDG.SSPRK3
```
## Surface
```@docs
TerraDG.BuffersFaceIntegral
TerraDG.rusanov
TerraDG.project_to_faces
TerraDG.evaluate_face_integral
```

## Volume
```@docs
TerraDG.BuffersVolume
TerraDG.evaluate_volume
```

## Filtering
```@docs
TerraDG.Filter
TerraDG.make_filter
TerraDG.IdentityFilter
TerraDG.evaluate_filter_scaling_matrix
TerraDG.requires_filtering
TerraDG.evaluate_filter_matrix
```

## Index
```@index
```
