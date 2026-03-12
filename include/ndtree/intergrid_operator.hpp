#ifndef AMR_INCLUDED_INTERGRID_OPERATOR
#define AMR_INCLUDED_INTERGRID_OPERATOR

#include "ndconcepts.hpp"
#include <array>
#include <concepts>
#include <iostream>

namespace amr::ndt::intergrid_operator
{

template <concepts::PatchLayout Patch_Layout>
struct linear_interpolator
{
    using patch_layout_t = Patch_Layout;
    using index_t        = typename patch_layout_t::index_t;
    using rank_t         = typename patch_layout_t::rank_t;

    template <std::integral auto N>
    static constexpr auto
        restriction(auto const& in, std::array<index_t, N> const& idxs) noexcept ->
        typename std::remove_cvref_t<decltype(in)>::value_type
    {
        using value_type = typename std::remove_cvref_t<decltype(in)>::value_type;
        value_type sum{};
        for (auto const idx : idxs)
        {
            // std::cout << idx << ": " << in[idx] << ' ';
            sum += in[idx];
        }
        // std::cout << " -> " << sum / static_cast<value_type>(N) << '\n';
        return sum / static_cast<value_type>(N);
    }
};

} // namespace amr::ndt::intergrid_operator

#endif
