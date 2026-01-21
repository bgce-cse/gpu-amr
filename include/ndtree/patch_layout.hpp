#ifndef AMR_INCLUDED_PATCH_LAYOUT
#define AMR_INCLUDED_PATCH_LAYOUT

#include "containers/container_concepts.hpp"
#include "containers/container_utils.hpp"
#include "containers/loop_control.hpp"
#include "containers/static_vector.hpp"
#include "ndconcepts.hpp"

namespace amr::ndt::patches
{

template <containers::concepts::StaticLayout Data_Layout, std::integral auto Halo_Width>
class patch_layout
{
public:
    using data_layout_t = Data_Layout;
    using size_type     = typename data_layout_t::size_type;

    using padded_layout_t = typename containers::utils::types::layout::padded_layout<
        data_layout_t>::template type<size_type{ Halo_Width * 2 }>;

    using shape_t = typename data_layout_t::shape_t;
    using index_t = typename data_layout_t::index_t;
    using rank_t  = typename data_layout_t::rank_t;

private:
    static constexpr auto      s_rank       = data_layout_t::rank();
    static constexpr size_type s_flat_size  = padded_layout_t::flat_size();
    static constexpr size_type s_halo_width = Halo_Width;

    [[nodiscard]]
    static constexpr auto full_iteration_impl() -> auto
    {
        constexpr auto start  = index_t{};
        constexpr auto end    = padded_layout_t::sizes();
        constexpr auto stride = index_t{ 1 };
        return containers::control::
            loop_control<typename padded_layout_t::shape_t, start, end, stride>{};
    }

    [[nodiscard]]
    static constexpr auto interior_iteration_impl() -> auto
    {
        static constexpr auto h = halo_width();

        constexpr auto                                       start  = s_halo_width;
        constexpr std::make_signed_t<decltype(s_halo_width)> end    = -s_halo_width;
        constexpr auto                                       stride = index_t{ 1 };
        return containers::control::
            loop_control<typename padded_layout_t::shape_t, start, end, stride>{};
    }

    template <concepts::Direction auto Direction>
    [[nodiscard]]
    static constexpr auto halo_iteration_control_impl() -> auto
    {
        using direction_t           = std::remove_cvref_t<decltype(Direction)>;
        static constexpr auto h     = halo_width();
        static constexpr auto rank  = direction_t::rank();
        static constexpr auto sizes = padded_layout_t::sizes();
        using vec_t                 = containers::static_vector<index_t, rank>;

        constexpr auto low       = Direction.is_negative();
        constexpr auto dimension = Direction.dimension();
        constexpr auto start     = []()
        {
            vec_t s{};
            for (rank_t i = 0; i != rank; ++i)
            {
                if (i != dimension)
                    s[i] = h;
                else
                    s[i] = low ? 0 : sizes[i] - h;
            }
            return s;
        }();
        constexpr auto end = []()
        {
            vec_t e{};
            for (rank_t i = 0; i != rank; ++i)
            {
                if (i != dimension)
                    e[i] = sizes[i] - h;
                else
                    e[i] = low ? h : sizes[i];
            }
            return e;
        }();
        return containers::control::
            loop_control<typename padded_layout_t::shape_t, start, end, index_t{ 1 }>{};
    }

public:
    using full_iteration_t     = decltype(full_iteration_impl());
    using interior_iteration_t = decltype(interior_iteration_impl());
    template <concepts::Direction auto Direction>
    using halo_iteration_control_t = decltype(halo_iteration_control_impl<Direction>());

    [[nodiscard]]
    static constexpr auto rank() noexcept -> rank_t
    {
        return s_rank;
    }

    [[nodiscard]]
    static constexpr auto halo_width() noexcept -> size_type
    {
        return s_halo_width;
    }

    [[nodiscard]]
    static constexpr auto flat_size() noexcept -> size_type
    {
        return s_flat_size;
    }
};

} // namespace amr::ndt::patches

#endif // AMR_INCLUDED_PATCH_LAYOUT
