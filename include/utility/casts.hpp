#ifndef INCLUDED_UTILITY_CASTS
#define INCLUDED_UTILITY_CASTS

#include "error_handling.hpp"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <type_traits>

namespace utility::casts
{

template <typename Dst, typename Src>
[[nodiscard]]
constexpr auto safe_cast(Src const& v) noexcept -> Dst
{
    constexpr auto is_same_type = std::is_same_v<Src, Dst>;
    constexpr auto is_ptr_to_ptr =
        std::is_pointer_v<Src> && std::is_pointer_v<Dst>;
    constexpr auto is_float_to_float =
        std::is_floating_point_v<Src> && std::is_floating_point_v<Dst>;
    constexpr auto is_number_to_number =
        std::is_arithmetic_v<Src> && std::is_arithmetic_v<Dst>;
    constexpr auto is_intptr_to_ptr = (std::is_same_v<std::uintptr_t, Src> ||
                                       std::is_same_v<std::intptr_t, Src>) &&
                                      std::is_pointer_v<Dst>;
    constexpr auto is_ptr_to_intptr =
        std::is_pointer_v<Src> && (std::is_same_v<std::uintptr_t, Dst> ||
                                   std::is_same_v<std::intptr_t, Dst>);
    if constexpr (is_same_type)
    {
        return v;
    }
    else if constexpr (is_intptr_to_ptr || is_ptr_to_intptr)
    {
        return reinterpret_cast<Dst>(v);
    }
    else if constexpr (is_ptr_to_ptr)
    {
        assert(dynamic_cast<Dst>(v) != nullptr);
        return static_cast<Dst>(v);
    }
    else if constexpr (is_float_to_float)
    {
        const auto casted = static_cast<Dst>(v);
        const auto casted_back = static_cast<Src>(casted);
        assert(!std::isnan(casted_back) && !std::isinf(casted_back));
        return casted;
    }
    else if constexpr (is_number_to_number)
    {
        const auto casted = static_cast<Dst>(v);
        const auto casted_back = static_cast<Src>(casted);
        assert(v == casted_back);
        return casted;
    }
    else
    {
        static_assert(std::false_type::value, "Cast Error");
        utility::error_handling::assert_unreachable();
    }
}

} // namespace utility::casts

#endif // INCLUDED_UTILITY_CASTS
