#ifndef AMR_INCLUDED_NDTREE
#define AMR_INCLUDED_NDTREE

#include <concepts>
#include <cstdint>
#include <type_traits>

namespace amr::ndt::tree
{

template <
    typename T,
    std::unsigned_integral auto Dim,
    std::unsigned_integral auto Fanout,
    std::unsigned_integral      Index_Type = std::uint32_t>
class ndtree
{
    static_assert(Dim > 1);
    static_assert(Fanout > 1);

public:
    using index_t       = Index_Type;
    using size_type     = index_t;
    using value_type    = T;
    using pointer       = value_type*;
    using const_pointer = value_type const*;
};

} // namespace amr::ndt::tree

#endif // AMR_INCLUDED_NDT_TREE
