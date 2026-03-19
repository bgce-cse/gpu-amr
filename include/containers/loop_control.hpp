#ifndef AMR_INCLUDED_LOOP_CONTROL
#define AMR_INCLUDED_LOOP_CONTROL

#include "container_concepts.hpp"
#include "utility/casts.hpp"
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
                std::array<index_t, s_rank> start{};
                for (auto i = rank_t{}; i != s_rank; ++i)
                    start[i] = static_cast<index_t>(
                        utility::casts::safe_cast<decltype(Start)>(s_sizes[i]) + Start
                    );
                return start;
            }
        }
        else
        {
            std::array<index_t, s_rank> start{};
            for (auto i = rank_t{}; i != s_rank; ++i)
                start[i] = index_t{ Start[i] };
            return start;
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
                std::array<index_t, s_rank> end{};
                for (auto i = rank_t{}; i != s_rank; ++i)
                    end[i] = static_cast<index_t>(
                        utility::casts::safe_cast<decltype(End)>(s_sizes[i]) + End
                    );
                return end;
            }
        }
        else
        {
            std::array<index_t, s_rank> end{};
            for (auto i = rank_t{}; i != s_rank; ++i)
                end[i] = index_t{ End[i] };
            return end;
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
    static constexpr auto at_idx(auto const& v, const rank_t idx) noexcept -> index_t
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
                (at_idx(s_end, i) <= index_t{ s_sizes[i] } &&
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
