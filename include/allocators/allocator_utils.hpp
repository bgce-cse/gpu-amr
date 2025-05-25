#ifndef AMR_INCLUDED_AMR_ALLOCATOR_UTILS
#define AMR_INCLUDED_AMR_ALLOCATOR_UTILS

#include <concepts>
#include <type_traits>

namespace amr::allocator::utils
{

[[nodiscard, gnu::const]]
consteval auto is_valid_alignment(std::unsigned_integral auto alignment) noexcept -> bool
{
    return alignment != 0 && ((alignment & (alignment - 1)) == 0);
}

} // namespace amr::allocator::utils

#endif // AMR_INCLUDED_AMR_ALLOCATOR_UTILS
