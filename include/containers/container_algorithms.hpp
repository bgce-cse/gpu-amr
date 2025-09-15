#ifndef AMR_INCLUDED_CONTAINER_FACTORIES
#define AMR_INCLUDED_CONTAINER_FACTORIES

#include "static_tensor.hpp"
#include "static_vector.hpp"
#include <concepts>
#include <utility>

namespace amr::containers::algorithms
{

namespace tensor
{

template <std::integral auto Rank, typename T, std::size_t N>
[[nodiscard]]
constexpr auto cartesian_expansion(std::array<T, N> const& v) noexcept -> auto
{
    using size_type = decltype(Rank);
    auto impl       = [&v]<std::size_t... Is>(std::index_sequence<Is...>) constexpr
        -> static_tensor<static_vector<T, Rank>, ((void)Is, size_type{ N })...>
    {
        using tensor_t =
            static_tensor<static_vector<T, Rank>, ((void)Is, size_type{ N })...>;
        auto ret = tensor_t::zero();
        auto idx = typename tensor_t::multi_index_t{};
        do
        {
            for (auto d = decltype(Rank){}; d != Rank; ++d)
            {
                ret[idx][d] = v[idx[d]];
            }
        } while (idx.increment());

        return ret;
    };
    return impl(std::make_index_sequence<Rank>{});
}

} // namespace tensor

} // namespace amr::containers::factory

#endif // AMR_INCLUDED_CONTAINER_FACTORIES
