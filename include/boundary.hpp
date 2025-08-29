#pragma once

#include "data_types.hpp"
#include "tree_types.hpp"
#include "utility/error_handling.hpp"
#include "utility/utility_concepts.hpp"

template <utility::concepts::Arithmetic T>
[[nodiscard]]
static constexpr auto neumann_bc(T const in_v, T const b_v = 0) noexcept -> T
{
    return in_v + b_v;
}
template <utility::concepts::Arithmetic T>
[[nodiscard]]
static constexpr auto dirichlet_bc(T const in_v, T const b_v = 0) noexcept -> T
{
    return T{2} * b_v - in_v;
}

enum struct BoundaryType
{
    None = 0,
    Dirichlet,
    Neumann,
};

struct BCSet
{
    BoundaryType u_bc_type;
    u_t::type u_bc_value;

    BoundaryType v_bc_type;
    v_t::type v_bc_value;

    BoundaryType p_bc_type;
    p_t::type p_bc_value;

    BoundaryType t_bc_type;
    t_t::type t_bc_value;
};

inline BCSet map_physical_bc(const PhysicalBC& bc)
{
    if (bc.type == "no_slip")
    {
        return {
            BoundaryType::Dirichlet,
            0.0,
            BoundaryType::Dirichlet,
            0.0,
            BoundaryType::Neumann,
            0.0,
            BoundaryType::Neumann,
            0.0
        };
    }
    else if (bc.type == "moving_wall")
    {
        return {
            BoundaryType::Dirichlet,
            bc.value,
            BoundaryType::Dirichlet,
            0.0,
            BoundaryType::Neumann,
            0.0,
            BoundaryType::Neumann,
            0.0
        };
    }
    else if (bc.type == "outlet")
    {
        return {
            BoundaryType::Neumann,
            0.0,
            BoundaryType::Neumann,
            0.0,
            BoundaryType::Dirichlet,
            0.0,
            BoundaryType::Neumann,
            0.0
        };
    }
    else if (bc.type == "inlet")
    {
        return {
            BoundaryType::Dirichlet,
            bc.value,
            BoundaryType::Dirichlet,
            0.0,
            BoundaryType::Neumann,
            0.0,
            BoundaryType::Neumann,
            0.0
        };
    }
    else if (bc.type == "temperature")
    {
        return {
            BoundaryType::Dirichlet,
            0.0,
            BoundaryType::Dirichlet,
            0.0,
            BoundaryType::Neumann,
            0.0,
            BoundaryType::Dirichlet,
            bc.value,
        };
    }
    throw std::runtime_error("Unknown BC type: ");
}

template <typename ValueType>
struct qoi_boundary
{
    using T = typename ValueType::type;
    using bc_t = T (*)(const T, const T);
    struct bc_config
    {
        BoundaryType type;
        T value;
    };
    struct domain_bc
    {
        bc_config top;
        bc_config left;
        bc_config right;
        bc_config bottom;
        bc_config internal;
    };

    qoi_boundary(domain_bc const& config) noexcept
        : _config{config}
        , _top_bc{config_fn(config.top.type)}
        , _left_bc{config_fn(config.left.type)}
        , _right_bc{config_fn(config.right.type)}
        , _bottom_bc{config_fn(config.bottom.type)}
        , _internal_bc{config_fn(config.internal.type)}
    {
    }

    [[nodiscard]]
    static constexpr auto config_fn(BoundaryType b) noexcept -> bc_t
    {
        switch (b)
        {
        case BoundaryType::Neumann:
            return neumann_bc;
        case BoundaryType::Dirichlet:
            return dirichlet_bc;
        case BoundaryType::None:
        default:
            return nullptr;
        }
    }

    [[nodiscard]]
    auto get_outter_boundary_value(
        auto const& tree, index_t morton_id, direction_t dir
    ) const noexcept
    {
        const auto internal_value =
            tree.template get_value_recur<ValueType>(morton_id);
        switch (dir)
        {
        case direction_t::N:
            return _top_bc(internal_value, _config.top.value);
        case direction_t::W:
            return _left_bc(internal_value, _config.left.value);
        case direction_t::E:
            return _right_bc(internal_value, _config.right.value);
        case direction_t::S:
            return _bottom_bc(internal_value, _config.bottom.value);
        case direction_t::NE:
            return std::midpoint(
                _bottom_bc(internal_value, _config.top.value),
                _bottom_bc(internal_value, _config.right.value)
            );
        case direction_t::SW:
            return std::midpoint(
                _bottom_bc(internal_value, _config.bottom.value),
                _bottom_bc(internal_value, _config.left.value)
            );
        case direction_t::SE:
            return std::midpoint(
                _bottom_bc(internal_value, _config.bottom.value),
                _bottom_bc(internal_value, _config.right.value)
            );
        case direction_t::NW:
            return std::midpoint(
                _bottom_bc(internal_value, _config.top.value),
                _bottom_bc(internal_value, _config.left.value)
            );
        default:
            utility::error_handling::assert_unreachable();
        }
    }

    [[nodiscard]]
    auto get_internal_boundary_value(
        auto const& tree,
        typename std::remove_cvref_t<decltype(tree)>::node_index_t node_id
    ) const noexcept
    {
        const auto internal_value =
            tree.template get_value_recur<ValueType>(node_id);
        return _internal_bc(internal_value, _config.internal.value);
    }

    domain_bc _config;
    bc_t _top_bc;
    bc_t _left_bc;
    bc_t _right_bc;
    bc_t _bottom_bc;
    bc_t _internal_bc;
};

struct domain_bc
{
    template <typename ValueType>
    [[nodiscard]]
    auto get_outter_boundary_value(
        auto const& tree, index_t morton_id, direction_t dir
    ) const noexcept
    {
        return get_bc<ValueType>().get_outter_boundary_value(
            tree, morton_id, dir
        );
    }

    template <typename ValueType>
    [[nodiscard]]
    auto get_internal_boundary_value(
        auto const& tree,
        std::remove_cvref_t<decltype(tree)>::node_index_t node_id
    ) const noexcept
    {
        return get_bc<ValueType>().get_internal_boundary_value(tree, node_id);
    }

    template <typename ValueType>
    auto get_bc() const -> auto&
    {
        if constexpr (std::is_same_v<u_t, ValueType> ||
                      std::is_same_v<f_t, ValueType>)
        {
            return u_bc;
        }
        if constexpr (std::is_same_v<v_t, ValueType> ||
                      std::is_same_v<g_t, ValueType>)
        {
            return v_bc;
        }
        if constexpr (std::is_same_v<p_t, ValueType>)
        {
            return p_bc;
        }
        if constexpr (std::is_same_v<t_t, ValueType>)
        {
            return t_bc;
        }
        if constexpr (std::is_same_v<rhs_t, ValueType>)
        {
            return _internal_rhs_bc_;
        }
        utility::error_handling::assert_unreachable();
    }

    qoi_boundary<u_t> u_bc;
    qoi_boundary<v_t> v_bc;
    qoi_boundary<p_t> p_bc;
    qoi_boundary<t_t> t_bc;
    qoi_boundary<rhs_t> _internal_rhs_bc_ =
        qoi_boundary<rhs_t>(typename qoi_boundary<rhs_t>::domain_bc(
            {BoundaryType::Neumann, 0}, {BoundaryType::Neumann, 0},
            {BoundaryType::Neumann, 0}, {BoundaryType::Neumann, 0},
            {BoundaryType::Neumann, 0}
        ));
};
