#ifndef AMR_INCLUDED_ALLOCATOR_UTILS
#define AMR_INCLUDED_ALLOCATOR_UTILS

#include <cassert>
#include <concepts>
#include <type_traits>

#ifndef NDEBUG
#    ifndef ALLOCATOR_DEBUG_INITIALIZE
#        define ALLOCATOR_DEBUG_INITIALIZE (true)
#    endif
#    if ALLOCATOR_DEBUG_INITIALIZE
#        include <algorithm>
#        define ALLOCATOR_DEBUG_INIT_VALUE    std::byte{ 0xCD }
#        define ALLOCATOR_DEBUG_ALLOC_VALUE   std::byte{ 0xAA }
#        define ALLOCATOR_DEBUG_DEALLOC_VALUE std::byte{ 0xDD }
#        define ALLOCATOR_DEBUG_RELEASE_VALUE std::byte{ 0xFE }
#    endif
#endif

namespace amr::allocator::utils
{

[[nodiscard, gnu::const]]
consteval auto is_valid_alignment(std::unsigned_integral auto alignment) noexcept -> bool
{
    return alignment != 0 && ((alignment & (alignment - 1)) == 0);
}

#if ALLOCATOR_DEBUG_INITIALIZE
static auto fill_buffer(
    std::byte* const         buffer,
    std::integral auto const size,
    std::byte const          value
)
{
    assert(size > 0);
    [[assume(size > 0)]];
    std::fill_n(buffer, size, value);
}
#endif

} // namespace amr::allocator::utils

#endif // AMR_INCLUDED_ALLOCATOR_UTILS
