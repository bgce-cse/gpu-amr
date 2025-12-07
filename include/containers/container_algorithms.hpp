#ifndef AMR_INCLUDED_CONTAINER_FACTORIES
#define AMR_INCLUDED_CONTAINER_FACTORIES

#include "container_utils.hpp"
#include "static_vector.hpp"
#include <concepts>

namespace amr::containers::algorithms
{

namespace tensor
{

template <std::integral auto Rank, typename T, std::integral auto Size>
[[nodiscard]]
constexpr auto cartesian_expansion(static_vector<T, Size> const& v) noexcept
    -> utils::types::tensor::hypercube_t<static_vector<T, Rank>, Size, Rank>
{
    using index_t     = typename std::remove_cvref_t<decltype(v)>::index_t;
    using hypercube_t = utils::types::tensor::
        hypercube_t<static_vector<T, index_t{ Rank }>, Size, index_t{ Rank }>;
    using multi_index_t = typename hypercube_t::multi_index_t;
    static_assert(std::is_same_v<typename multi_index_t::index_t, index_t>);
    auto ret = hypercube_t{};
    auto idx = multi_index_t{};
    do
    {
        for (index_t d{}; d != index_t{ Rank }; ++d)
        {
            ret[idx][d] = v[idx[d]];
        }
    } while (idx.increment());

    return ret;
}

template <concepts::StaticContainer T1, concepts::StaticContainer T2>
[[nodiscrad]]
constexpr auto tensor_product(T1 const& a, T2 const& b) noexcept -> utils::types::tensor::
    tensor_product_result_t<std::remove_cvref_t<T1>, std::remove_cvref_t<T2>>
{
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
