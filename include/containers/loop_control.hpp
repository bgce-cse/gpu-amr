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
    static constexpr auto s_rank   = shape_t::rank();
    static constexpr auto s_sizes  = shape_t::sizes();
    static constexpr auto s_stride = index_t{ Stride };
    static constexpr auto s_start  = []
    {
        using start_t = std::remove_cvref_t<decltype(Start)>;
        if constexpr (std::is_arithmetic_v<start_t>)
        {
            if constexpr (Start >= 0)
            {
                return index_t{ Start };
            }
            else
            {
                auto sizes = s_sizes;
                for (auto& e : sizes)
                    e += Start;
                return sizes;
            }
        }
        else
        {
            return index_t{ Start };
        }
    }();
    static constexpr auto s_end = []
    {
        using end_t = std::remove_cvref_t<decltype(End)>;
        if constexpr (std::is_arithmetic_v<end_t>)
        {
            if constexpr (End >= 0)
            {
                return index_t{ End };
            }
            else
            {
                auto sizes = s_sizes;
                for (auto& e : sizes)
                    e += End;
                return sizes;
            }
        }
        else
        {
            return index_t{ End };
        }
    }();

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
        static_assert(check_param(s_start));
        static_assert(check_param(s_end));
        static_assert(check_param(s_stride));
        for (auto i = decltype(s_rank){}; i != s_rank; ++i)
        {
            if ((at_idx(s_start, i) >= index_t{}) &&
                (at_idx(s_start, i) <= at_idx(s_end, i)) &&
                (at_idx(s_end, i) <= s_sizes[i] &&
                 ((at_idx(s_end, i) - at_idx(s_start, i)) % at_idx(s_stride, i) ==
                  index_t{})))
            {
                continue;
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
        return at_idx(s_start, i);
    }

    [[nodiscard]]
    static constexpr auto end(const rank_t i) noexcept -> index_t
    {
        return at_idx(s_end, i);
    }

    [[nodiscard]]
    static constexpr auto stride(const rank_t i) noexcept -> index_t
    {
        return at_idx(s_stride, i);
    }
};

} // namespace amr::containers::control

#endif // AMR_INCLUDED_LOOP_CONTROL
