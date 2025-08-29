#pragma once

#include "Datastructures.hpp"
#include "boundary.hpp"
#include "data_types.hpp"
#include "refinement_kernels.hpp"
#include "tree_types.hpp"
#include "utility/error_handling.hpp"
#include <algorithm>
#include <cassert>
#include <ranges>
#include <variant>

class sim_domain
{
  public:
    using tree_t = ndtree_t<sim_domain>;
    using bc_t = domain_bc;
    using nephews_t = typename tree_t::neighbours_t::nephews;
    using sibling_t = typename tree_t::neighbours_t::sibling;
    using uncle_t = typename tree_t::neighbours_t::uncle;

  public:
    sim_domain(
        std::size_t capacity, cell const& initial_value,
        unsigned int initial_refinement, double x_length, double y_length,
        bc_t bc_config, Matrix<cell_type> const* const p_topology = nullptr
    ) noexcept
        : _bc(bc_config)
        , _tree(
              this, capacity,
              topology_fitting_kernel<tree_t>(initial_refinement, p_topology),
              initial_value
          )
        , _x_length(x_length)
        , _y_length(y_length)
    {
    }

    template <typename ValueType>
    auto get_value(index_t morton_id) const -> auto
    {
        assert(_tree.get_cell_type(morton_id) == cell_type::FLUID);
        return _tree.template get<ValueType>(morton_id);
    }
    template <typename ValueType>
    auto get_value(index_t morton_id) -> auto&
    {
        assert(_tree.get_cell_type(morton_id) == cell_type::FLUID);
        return _tree.template get<ValueType>(morton_id);
    }

    template <typename ValueType>
    auto restrict_impl(std::ranges::input_range auto&& values) const ->
        typename ValueType::type
    {
        using ret_t = typename ValueType::type;
        ret_t sum = {};
        std::size_t count = 0;

        for (auto const& val : values)
        {
            sum += val;
            ++count;
        }
        return count != 0 ? (sum / ret_t(count)) : ret_t{};
    }

    /**
     * @brief Interpolate implementation to obtain the value of a virtual cell
     * defined by a parent id and a child offset.
     *
     * @tparam ValueType QOI to interpolate
     * @param morton_id PArent id
     * @param offset Child offset
     * @return Interpolated value
     */
    template <typename ValueType>
    auto interpolate_impl(
        index_t morton_id, typename index_t::offset_t offset
    ) const
    {
        direction_t left_side;
        direction_t right_side;
        switch (offset)
        {
        case 0:
            left_side = direction_t::W;
            right_side = direction_t::N;
            break;
        case 1:
            left_side = direction_t::N;
            right_side = direction_t::E;
            break;
        case 2:
            left_side = direction_t::S;
            right_side = direction_t::W;
            break;
        case 3:
            left_side = direction_t::E;
            right_side = direction_t::S;
            break;
        default:
            utility::error_handling::assert_unreachable();
            break;
        }

        const auto parent_value =
            get_value_or_internal_bound<ValueType>(morton_id, morton_id);
        const auto left_side_value =
            get_neighbor_value<ValueType>(morton_id, left_side);
        const auto right_side_value =
            get_neighbor_value<ValueType>(morton_id, right_side);

        return 0.5 * parent_value + 0.25 * left_side_value +
               0.25 * right_side_value;
    }

    /**
     * @brief Computes the possibly virtual neigbouring value of a node in a
     * certain direction.
     *
     * @tparam ValueType QOI to compute.
     * @param neighbours Neigbours from which to interpolate.
     * @param morton_id Requesting node. Needed to compute boundary values.
     * @param dir Direction in which the values are requested.
     * @return
     */
    template <typename ValueType>
    auto interpolate(
        typename tree_t::neighbour_alternatives_t const& neighbours,
        index_t morton_id, direction_t dir
    ) const -> typename ValueType::type
    {
        using ret_t = typename ValueType::type;
        if (std::holds_alternative<sibling_t>(neighbours))
        {
            const auto& s = std::get<sibling_t>(neighbours);
            return get_value_or_internal_bound<ValueType>(s.value, morton_id);
        }
        else if (std::holds_alternative<nephews_t>(neighbours))
        {
            auto const& n = std::get<nephews_t>(neighbours);
            return restrict_impl<ValueType>(
                n.values |
                std::views::transform(
                    [this, morton_id](auto const node_idx) -> ret_t {
                        return get_value_or_internal_bound<ValueType>(
                            node_idx, morton_id
                        );
                    }
                )
            );
        }
        else if (std::holds_alternative<uncle_t>(neighbours))
        {
            auto const& u = std::get<uncle_t>(neighbours);

            const auto uncle_value =
                get_value_or_internal_bound<ValueType>(u.value, morton_id);
            if (_tree.get_cell_type(u.value) == cell_type::OBSTACLE)
            {
                return uncle_value;
            }
            const auto offset = index_t::offset_of(morton_id);
            typename index_t::offset_t cousin_offset;
            switch (offset)
            {
            case 0:
                switch (dir)
                {
                case direction_t::N:
                    cousin_offset = 2;
                    break;
                case direction_t::W:
                    cousin_offset = 1;
                    break;
                case direction_t::NW:
                case direction_t::SW:
                case direction_t::NE:
                case direction_t::SE:
                case direction_t::S:
                case direction_t::E:
                default:
                    utility::error_handling::assert_unreachable();
                    break;
                }
                break;
            case 1:
                switch (dir)
                {
                case direction_t::N:
                    cousin_offset = 3;
                    break;
                case direction_t::E:
                    cousin_offset = 0;
                    break;
                case direction_t::NW:
                case direction_t::NE:
                case direction_t::SE:
                case direction_t::SW:
                case direction_t::S:
                case direction_t::W:
                default:
                    utility::error_handling::assert_unreachable();
                    break;
                }
                break;
            case 2:
                switch (dir)
                {
                case direction_t::S:
                    cousin_offset = 0;
                    break;
                case direction_t::W:
                    cousin_offset = 3;
                    break;
                case direction_t::NW:
                case direction_t::SW:
                case direction_t::SE:
                case direction_t::NE:
                case direction_t::E:
                case direction_t::N:
                default:
                    utility::error_handling::assert_unreachable();
                    break;
                }
                break;
            case 3:
                switch (dir)
                {
                case direction_t::S:
                    cousin_offset = 1;
                    break;
                case direction_t::E:
                    cousin_offset = 2;
                    break;
                case direction_t::NE:
                case direction_t::SE:
                case direction_t::SW:
                case direction_t::NW:
                case direction_t::N:
                case direction_t::W:
                default:
                    utility::error_handling::assert_unreachable();
                    break;
                }
                break;
            default:
                utility::error_handling::assert_unreachable();
            }
            return interpolate_impl<ValueType>(u.value, cousin_offset);
        }
        else
        {
            utility::error_handling::assert_unreachable();
        }
    }

    /**
     * @brief Computes the value of a virtual neighbour at the same level.
     *
     * @tparam ValueType QOI
     * @param morton_id Source node
     * @param dir Direction
     * @return The possibly manufactured value
     */
    template <typename ValueType>
    auto get_neighbor_value(index_t morton_id, direction_t dir) const ->
        typename ValueType::type
    {
        const auto neighbors_opt = _tree.get_neighbors(morton_id, dir);
        if (!neighbors_opt)
        {
            return _bc.template get_outter_boundary_value<ValueType>(
                _tree, morton_id, dir
            );
        }

        const auto& neighbors = *neighbors_opt;
        return interpolate<ValueType>(neighbors, morton_id, dir);
    }

    [[nodiscard]]
    auto tree() const -> auto const&
    {
        return _tree;
    }
    [[nodiscard]]
    auto tree() -> auto&
    {
        return _tree;
    }

    auto dx(index_t morton_id)
    {
        auto [coord, level] = index_t::decode(morton_id);
        auto cell_length = std::pow(2.0, -level);
        return _x_length * cell_length;
    }

    auto dy(index_t morton_id)
    {
        auto [coord, level] = index_t::decode(morton_id);
        auto cell_length = std::pow(2.0, -level);
        return _y_length * cell_length;
    }

    [[nodiscard]]
    double x_length() const
    {
        return _x_length;
    }

    [[nodiscard]]
    double y_length() const
    {
        return _y_length;
    }

  private:
    /**
     * @brief Accesses a value inside the domain, which can be an internal bound
     * or a value stred in the tree.
     *
     * @tparam ValueType QOI
     * @param morton_id Node index to access
     * @param morton_id_og Requesting node id
     * @return The possibly computed resquested value
     */
    template <typename ValueType>
    auto get_value_or_internal_bound(
        index_t morton_id, index_t morton_id_og
    ) const
    {
        auto iter = _tree.find_index(morton_id);
        if (iter)
        {
            auto idx = _tree.index_at(morton_id);
            return _tree.get_cell_type(idx) == cell_type::FLUID
                       ? _tree.template get<ValueType>(idx)
                       : _bc.get_internal_boundary_value<ValueType>(
                             _tree, morton_id_og
                         );
        }

        return _tree.template get_value_recur<ValueType>(morton_id);
    }

  private:
    bc_t _bc;
    tree_t _tree;
    double _x_length;
    double _y_length;
};
