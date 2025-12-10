#ifndef AMR_INCLUDED_NDUTILS
#define AMR_INCLUDED_NDUTILS

#include "utility/constexpr_functions.hpp"
#include <cassert>
#include <concepts>
#include <type_traits>

namespace amr::ndt::utils
{

[[nodiscard]]
consteval auto
    compute_nd_fanout(std::integral auto dim, std::integral auto fanout) noexcept
    -> std::common_type_t<decltype(dim), decltype(fanout)>
{
    using return_t = std::common_type_t<decltype(dim), decltype(fanout)>;
    return utility::cx_functions::pow(return_t{ dim }, return_t{ fanout });
}

template <class... Ts>
struct overloads : Ts...
{
    using Ts::operator()...;
};

} // namespace amr::ndt::utils

#endif // AMR_INCLUDED_NDUTILS
