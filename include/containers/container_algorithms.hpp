#ifndef AMR_INCLUDED_CONTAINER_FACTORIES
#define AMR_INCLUDED_CONTAINER_FACTORIES

#include "container_utils.hpp"
#include "static_tensor.hpp"
#include "static_vector.hpp"
#include <concepts>
#include <utility>

namespace amr::containers::algorithms
{

namespace tensor
{

template <std::size_t Rank, typename T, std::integral auto Size>
[[nodiscard]]
constexpr auto cartesian_expansion(static_vector<T, Size> const& v) noexcept
    -> utils::types::tensor::hypercube_t<static_vector<T, Rank>,0, Size, Rank>
{
    using hypercube_t =
        utils::types::tensor::hypercube_t<static_vector<T, Rank>,0, Size, Rank>;
    using multi_index_t = typename hypercube_t::multi_index_t;
    using index_t       = typename multi_index_t::index_t;
    auto ret            = hypercube_t{};
    auto idx            = multi_index_t{};
    do
    {
        for (index_t d{}; d != index_t{ Rank }; ++d)
        {
            ret[idx][d] = v[idx[d]];
        }
    } while (idx.increment());

    return ret;
}

template <
    std::floating_point F,
    std::integral auto  Rank,
    std::integral auto  Order,
    std::integral auto  Dofs>
[[nodiscard]]
constexpr auto evaluate_basis(
    utils::types::tensor::hypercube_t<static_vector<F, Dofs>, Order, Rank> const& coeffs,
    static_vector<F, Rank> const&                                                 x,
    static_vector<F, Order> const& quad_points
) noexcept -> F
{
    // TODO
}

} // namespace tensor

} // namespace amr::containers::algorithms

#endif // AMR_INCLUDED_CONTAINER_FACTORIES
