#ifndef AMR_INCLUDED_BOUNDARY_CONDITIONS
#define AMR_INCLUDED_BOUNDARY_CONDITIONS

#include <array>
#include <cstddef>

namespace amr::ndt::solver
{

enum class bc_type : std::uint8_t
{
    Periodic,
    Dirichlet,
    Neumann,
    Extrapolate
};

template<typename PhysicsSystem, typename CellType>
struct boundary_condition_set
{
    static constexpr std::size_t n_dimensions = PhysicsSystem::n_dimension;
    static constexpr std::size_t n_directions = 2 * n_dimensions;
    static constexpr std::size_t n_fields = CellType::n_fields;

    using bc_type_array_t = std::array<std::array<bc_type, n_directions>, n_fields>;
    bc_type_array_t bc_types;
    
    std::array<std::array<double, n_directions>, n_fields> bc_values;
    
    constexpr boundary_condition_set() 
    {
      
        for (std::size_t i = 0; i < n_fields; ++i) {
            bc_types[i].fill(bc_type::Extrapolate);
            bc_values[i].fill(0.0);
        }
    }
    
    
    template<typename FieldType>
    constexpr bc_type get_bc_type(std::size_t direction) const noexcept
    {
        constexpr std::size_t field_idx = FieldType::index();
        return bc_types[field_idx][direction];
    }
    
    
    template<typename FieldType>
    constexpr double get_bc_value(std::size_t direction) const noexcept
    {
        constexpr std::size_t field_idx = FieldType::index();
        return bc_values[field_idx][direction];
    }
    

    template<typename FieldType>
    constexpr void set_bc(std::size_t direction, bc_type type, double value = 0.0) noexcept
    {
        constexpr std::size_t field_idx = FieldType::index();
        bc_types[field_idx][direction] = type;
        bc_values[field_idx][direction] = value;
    }
    
    template<typename FieldType>
    constexpr void set_bc_all(bc_type type, double value = 0.0) noexcept
    {
        constexpr std::size_t field_idx = FieldType::index();
        bc_types[field_idx].fill(type);
        bc_values[field_idx].fill(value);
    }
};

} // namespace amr::ndt::solver

#endif