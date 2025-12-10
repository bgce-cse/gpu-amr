#ifndef AMR_INCLUDED_LOOP_CONTROL
#define AMR_INCLUDED_LOOP_CONTROL

#include "container_concepts.hpp"
#include "utility/error_handling.hpp"
#include <ranges>

namespace amr::containers::control
{

template <concepts::StaticShape S, auto Start, auto End, auto Stride>
class loop_control
{
public:
    using shape_t = S;
    using index_t = typename S::index_t;
    using rank_t  = typename S::rank_t;

private:
    static constexpr auto s_rank = shape_t::rank();

public:
    [[nodiscard]]
    static constexpr auto rank() noexcept -> rank_t
    {
        return shape_t::rank();
    }

private:
    [[nodiscard]]
    static constexpr auto at_idx(auto const& v, const rank_t idx) noexcept
        -> decltype(auto)
    {
        using v_t = std::remove_cvref_t<decltype(v)>;
        if constexpr (std::is_arithmetic_v<v_t>)
        {
            return v;
        }
        else if constexpr (std::ranges::range<v_t>)
        {
            using size_type = typename v_t::size_type;
            return v[static_cast<size_type>(idx)];
        }
        else
        {
            utility::error_handling::assert_unreachable();
        }
    };

    [[nodiscard]]
    static consteval auto check_param(auto const& param) noexcept -> bool
    {
        using param_t = std::remove_cvref_t<decltype(param)>;
        static_assert(std::is_arithmetic_v<param_t> || std::ranges::range<param_t>);
        if constexpr (std::is_arithmetic_v<param_t>)
        {
            return true;
        }
        else if constexpr (std::ranges::range<param_t>)
        {
            return std::ranges::size(param) == s_rank;
        }
        else
        {
            return false;
        }
    }

    [[nodiscard]]
    static consteval auto is_valid() noexcept -> bool
    {
        static_assert(check_param(Start));
        static_assert(check_param(Stride));
        static_assert(check_param(End));
        for (auto i = decltype(s_rank){}; i != s_rank; ++i)
        {
            if ((at_idx(Start, i) >= index_t{}) && (at_idx(Start, i) <= at_idx(End, i)) &&
                (at_idx(End, i) <= shape_t::size(i)) &&
                ((at_idx(End, i) - at_idx(Start, i)) % at_idx(Stride, i) == index_t{}))
            {
            }
            else
            {
                return false;
            }
        }
        return true;
    }

    static_assert(is_valid());

public:
    [[nodiscard]]
    static constexpr auto start(const rank_t i) noexcept -> index_t
    {
        return at_idx(Start, i);
    }

    [[nodiscard]]
    static constexpr auto end(const rank_t i) noexcept -> index_t
    {
        return at_idx(End, i);
    }

    [[nodiscard]]
    static constexpr auto stride(const rank_t i) noexcept -> index_t
    {
        return at_idx(Stride, i);
    }
};

} // namespace amr::containers::control

#endif // AMR_INCLUDED_LOOP_CONTROL
