#pragma once

#include "Datastructures.hpp"
#include "data_types.hpp"
#include "utility/operations.hpp"
#include <iostream>

template <typename Tree>
class topology_fitting_kernel
{
  public:
    using refine_status_t = typename Tree::refine_status_t;
    using index_t = typename Tree::node_index_t;

  public:
    topology_fitting_kernel(
        unsigned int initial_refinement,
        Matrix<cell_type> const* const p_topology
    )
        : _initial_refinement(initial_refinement)
        , _p_topology{p_topology}
    {
        assert(initial_refinement < index_t::max_depth());
    }

    [[nodiscard]]
    refine_status_t operator()(index_t const& node_idx)
    {
        const auto level = index_t::level(node_idx);
        const auto coords = index_t::rel_coords(node_idx);
        auto refine = false;
        if (level < _initial_refinement)
        {
            refine = true;
        }
        else if (level >= index_t::max_depth())
        {
            refine = false;
        }
        else if (_p_topology)
        {
            const auto rows = (unsigned)_p_topology->num_rows();
            assert(utility::basic_operations::is_pow2(rows));
            assert(rows == (unsigned)_p_topology->num_cols());
            const auto topology_depth = (unsigned int)std::log2(rows);
            const auto topology_size = rows;
            if (level >= topology_depth)
            {
                refine = false;
            }
            else
            {
                const auto x =
                    (unsigned)std::round(coords[0] * (float)topology_size);
                const auto y =
                    (unsigned)std::round(coords[1] * (float)topology_size);
                const auto grain_size =
                    (unsigned int)std::pow(2, topology_depth - level);

                const auto first = _p_topology->operator()(x, y);
                for (auto i = 0u; i != grain_size; ++i)
                {
                    for (auto j = 0u; j != grain_size; ++j)
                    {
                        if (_p_topology->operator()(x + i, y + j) != first)
                        {
                            refine = true;
                            goto end;
                        }
                    }
                }
            }
        }
    end:
        _reapply |= refine;
        return refine ? refine_status_t::Refine : refine_status_t::Stable;
    }

    [[nodiscard]]
    bool reapply() noexcept
    {
        const auto ret = _reapply;
        _reapply = false;
        return ret;
    }

    const Matrix<cell_type>* pgm_data() const
    {
        return _p_topology;
    }

  private:
    unsigned int _initial_refinement;
    Matrix<cell_type> const* _p_topology;
    bool _reapply = true;
};
