#ifndef AMR_INCLUDED_CONTAINER_TRAITS
#define AMR_INCLUDED_CONTAINER_TRAITS

#include <ranges>

namespace amr::containers::traits
{

namespace extent
{

template <std::ranges::range R>
constexpr auto extent_of() noexcept -> typename R::size_type
{
    if constexpr (R::rank())
    {
        return R::rank();
    }
    else if constexpr (R::size())
    {
        return R::size();
    }
    else if constexpr (std::declval<R&>().size())
    {
        return std::declval<R&>().size();
    }
    else
    {
        static_assert(false);
    }
}

} // namespace extent

} // namespace amr::containers::concepts

#endif // AMR_INCLUDED_CONTAINER_TRAITS
