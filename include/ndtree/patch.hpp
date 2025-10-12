#ifndef AMR_INCLUDED_PATCH
#define AMR_INCLUDED_PATCH

#include "containers/container_utils.hpp"
#include "containers/static_tensor.hpp"
#include "ndconcepts.hpp"
#include "ndutils.hpp"
#include "utility/constexpr_functions.hpp"
#include <concepts>

namespace amr::ndt::patches
{

template <
    typename T,
    containers::concepts::StaticLayout Data_Layout,
    std::integral auto                 Fanout,
    std::integral auto                 Halo_Width>
class patch
{
public:
    using data_layout_t   = Data_Layout;
    using padded_layout_t = typename containers::utils::types::layout::padded_layout<
        data_layout_t>::type<Halo_Width * 2>;

    using value_type      = std::remove_cv_t<T>;
    using size_type       = typename data_layout_t::size_type;
    using const_iterator  = value_type const*;
    using iterator        = value_type*;
    using const_reference = value_type const&;
    using reference       = value_type&;

    using index_t              = typename data_layout_t::index_t;
    using rank_t               = typename data_layout_t::rank_t;
    using padded_multi_index_t = typename padded_layout_t::multi_index_t;
    using container_t          = containers::static_tensor<value_type, padded_layout_t>;

private:
    static constexpr auto      s_dim       = data_layout_t::rank();
    static constexpr size_type s_1d_fanout = Fanout;
    static constexpr size_type s_nd_fanout =
        utility::cx_functions::pow(s_1d_fanout, s_dim);
    static constexpr size_type s_halo_width = Halo_Width;

    static_assert(s_nd_fanout > 1);
    static_assert(
        utils::patches::multiples_of(data_layout_t::sizes(), s_1d_fanout),
        "All patch dimensions must be multiples of the fanout"
    );

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
    static constexpr auto fanout() noexcept -> size_type
    {
        return s_1d_fanout;
    }

    [[nodiscard]]
    static constexpr auto nd_fanout() noexcept -> size_type
    {
        return s_nd_fanout;
    }

    [[nodiscard]]
    constexpr auto data() noexcept -> container_t&
    {
        return m_data;
    }

    [[nodiscard]]
    constexpr auto data() const noexcept -> container_t const&
    {
        return m_data;
    }

    template <class... I>
        requires(sizeof...(I) == s_dim) && (std::integral<std::remove_cvref_t<I>> && ...)
    [[nodiscard]]
    constexpr auto operator[](I&&... idxs) const noexcept -> const_reference
    {
        return m_data[std::forward<decltype(idxs)>(idxs)...];
    }

    template <class... I>
        requires(sizeof...(I) == s_dim) && (std::integral<std::remove_cvref_t<I>> && ...)
    [[nodiscard]]
    constexpr auto operator[](I&&... idxs) noexcept -> reference
    {
        return const_cast<reference>(
            std::as_const(*this).operator[](std::forward<decltype(idxs)>(idxs)...)
        );
    }

    [[nodiscard]]
    constexpr auto operator[](padded_multi_index_t const& multi_idx) const noexcept
        -> const_reference
    {
        return m_data[std::forward<decltype(multi_idx)>(multi_idx)];
    }

    [[nodiscard]]
    constexpr auto operator[](padded_multi_index_t const& multi_idx) noexcept -> reference
    {
        return const_cast<reference>(std::as_const(*this).operator[](multi_idx));
    }

    [[nodiscard]]
    constexpr auto operator[](index_t const linear_idx) const noexcept -> const_reference
    {
        return m_data[linear_idx];
    }

    [[nodiscard]]
    constexpr auto operator[](index_t const linear_idx) noexcept -> reference
    {
        return const_cast<reference>(std::as_const(*this).operator[](linear_idx));
    }

private:
    container_t m_data;
};

} // namespace amr::ndt::patches

#endif // AMR_INCLUDED_PATCH
