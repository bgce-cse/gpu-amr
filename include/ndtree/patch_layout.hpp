#ifndef AMR_INCLUDED_PATCH_LAYOUT
#define AMR_INCLUDED_PATCH_LAYOUT

#include "containers/container_concepts.hpp"
#include "containers/container_utils.hpp"

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

    using index_t = typename data_layout_t::index_t;
    using rank_t  = typename data_layout_t::rank_t;

private:
    static constexpr auto      s_dim        = data_layout_t::rank();
    static constexpr size_type s_flat_size  = padded_layout_t::flat_size();
    static constexpr size_type s_halo_width = Halo_Width;

public:
    [[nodiscard]]
    static constexpr auto dimension() noexcept -> rank_t
    {
        return s_dim;
    }

    [[nodiscard]]
    static constexpr auto halo_width() noexcept -> size_type
    {
        return s_halo_width;
    }

    [[nodiscard]]
    static constexpr auto flat_size() noexcept -> rank_t
    {
        return s_flat_size;
    }
};

} // namespace amr::ndt::patches

#endif // AMR_INCLUDED_PATCH_LAYOUT
