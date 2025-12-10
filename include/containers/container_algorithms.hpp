#ifndef AMR_INCLUDED_CONTAINER_FACTORIES
#define AMR_INCLUDED_CONTAINER_FACTORIES

#include "container_manipulations.hpp"
#include "container_utils.hpp"
#include "static_vector.hpp"
#include <concepts>

namespace amr::containers::algorithms
{

namespace tensor
{

// TODO: Update to shaped for rather than multiindex iteration
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
[[nodiscard]]
constexpr auto tensor_product(T1 const& t1, T2 const& t2) noexcept -> utils::types::
    tensor::tensor_product_result_t<std::remove_cvref_t<T1>, std::remove_cvref_t<T2>>
{
    using ret_t = utils::types::tensor::
        tensor_product_result_t<std::remove_cvref_t<T1>, std::remove_cvref_t<T2>>;
    ret_t ret{};
    using lc_t = control::loop_control<ret_t, 0, ret_t::sizes(), 1>;
    static_assert(
        std::tuple_size_v<std::remove_cvref_t<decltype(ret_t::sizes())>> ==
        T1::rank() + T2::rank()
    );
    manipulators::shaped_for<lc_t>(
        [&ret](auto const& a, auto const& b, auto const& idxs)
        {
            ret[idxs] = a[std::span<typename ret_t::index_t const, T1::rank()>{
                            idxs.data(), T1::rank() }] *
                        b[std::span<typename ret_t::index_t const, T2::rank()>{
                            idxs.data() + T1::rank(), T2::rank() }];
        },
        t1,
        t2
    );
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
