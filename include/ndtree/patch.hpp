#ifndef AMR_INCLUDED_PATCH
#define AMR_INCLUDED_PATCH

#include "containers/container_utils.hpp"
#include "containers/static_tensor.hpp"
#include "ndconcepts.hpp"
#include <concepts>

namespace amr::ndt::patches
{

template <
    typename T,
    containers::concepts::StaticLayout Data_Layout,
    std::integral auto                 Halo_Width>
class patch
{
    using data_layout_t = Data_Layout;
    using patch_layout_t =
        typename utils::layout::padded_layout<data_layout_t>::type<Halo_Width * 2>;

    using value_type      = std::remove_cv_t<T>;
    using size_type       = data_layout_t::size_type;
    using const_iterator  = value_type const*;
    using iterator        = value_type*;
    using const_reference = value_type const&;
    using reference       = value_type&;

    using index_t     = typename data_layout_t::indext_t;
    using container_t = containers::static_tensor<value_type, patch_layout_t>;

    inline static constexpr auto s_halo_width = Halo_Width;

    template <typename Multi_Index>
    [[nodiscard]]
    constexpr static auto to_multi_index(index_t const linear_index) noexcept
        -> Multi_Index
    {
        using multi_index_t = Multi_Index;
        using rank_t        = typename multi_index_t::rank_t;
        using size_type     = typename multi_index_t::size_type;
        using index_t       = typename multi_index_t::index_t;

        constexpr auto rank = multi_index_t::rank();
        assert(linear_index < multi_index_t::elements());
        if constexpr (std::is_signed_v<size_type>)
        {
            assert(linear_index >= 0);
        }

        multi_index_t multi_idx{};

        if constexpr (std::is_same_v<Multi_Index, typename patch_layout_t::multi_index_t>)
        {
            using layout_t = patch_layout_t;
            auto remainder = linear_index;
            for (rank_t d = 0; d < multi_index_t::s_rank; ++d)
            {
                const auto stride = layout_t::s_strides[d];
                multi_idx[d]      = static_cast<index_t>(remainder / stride);
                remainder %= stride;
            }
        }
        else if constexpr (std::is_same_v<
                               Multi_Index,
                               typename data_layout_t::multi_index_t>){
        }

            return multi_idx;
    }

private:
    container_t m_data;
};

} // namespace amr::ndt::patches

#endif // AMR_INCLUDED_PATCH
