#ifndef AMR_INCLUDED_NDUTILS
#define AMR_INCLUDED_NDUTILS

#include "utility/constexpr_functions.hpp"
#include <concepts>
#include <type_traits>

namespace amr::ndt::utils
{

consteval auto subdivisions(
    std::unsigned_integral auto dim,
    std::unsigned_integral auto fanout
) noexcept -> std::common_type_t<decltype(dim), decltype(fanout)>
{
    return utility::cx_functions::pow(dim, fanout);
}

} // namespace amr::ndt::utils

#endif // AMR_INCLUDED_NDUTILS
