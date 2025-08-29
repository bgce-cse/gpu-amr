#ifndef AMR_INLUDED_STATIC_TENSOR
#define AMR_INLUDED_STATIC_TENSOR

#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <type_traits>
#include "utility/utility_concepts.hpp"

#ifndef NDEBUG
#    define AMR_CONTANERS_CHECKBOUNDS
#endif

namespace amr::containers
{

template <typename T, std::integral auto N, std::integral auto... Ns>
    requires utility::concepts::are_same<decltype(N), decltype(Ns)...> && (N > 0) && ((Ns > 0) && ...)
class static_tensor
{
public:
    using value_type      = T;
    using size_type       = std::common_type_t<decltype(N), decltype(Ns)...>;
    using index_t         = size_type;
    using const_iterator  = value_type const*;
    using iterator        = value_type*;
    using const_reference = value_type const&;
    using reference       = value_type&;

    inline static constexpr size_type s_dims = sizeof...(Ns) + 1;
    inline static constexpr auto s_sizes     = std::array<size_type, s_dims>{ { N, Ns... } };
    inline static constexpr size_type s_flat_size =
        std::ranges::fold_left(s_sizes, size_type{ 1 }, std::multiplies{});

    static_assert(std::is_trivially_copyable_v<T>);
    static_assert(std::is_standard_layout_v<T>);
    static_assert(s_dims > 0);

    [[nodiscard]]
    constexpr static auto flat_size() noexcept -> size_type
    {
        return s_flat_size;
    }

    [[nodiscard]]
    constexpr static auto size(index_t const i) noexcept -> size_type
    {
        assert(i < s_dims);
        return s_sizes[i];
    }
};

} // namespace amr::containers

#endif // AMR_INCLUDED_STATIC_TENSOR
