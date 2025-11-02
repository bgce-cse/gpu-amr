#ifndef AMR_INCLUDED_PATCH
#define AMR_INCLUDED_PATCH

#include "containers/container_manipulations.hpp"
#include "containers/static_tensor.hpp"
#include "containers/static_vector.hpp"
#include "ndconcepts.hpp"
#include <concepts>

namespace amr::ndt::patches
{

template <typename T, concepts::PatchLayout Patch_Layout>
class patch
{
public:
    using patch_layout_t  = Patch_Layout;
    using data_layout_t   = typename patch_layout_t::data_layout_t;
    using padded_layout_t = typename patch_layout_t::padded_layout_t;

    using value_type      = std::remove_cv_t<T>;
    using size_type       = typename patch_layout_t::size_type;
    using const_iterator  = value_type const*;
    using iterator        = value_type*;
    using const_reference = value_type const&;
    using reference       = value_type&;

    using index_t              = typename patch_layout_t::index_t;
    using rank_t               = typename patch_layout_t::rank_t;
    using padded_multi_index_t = typename padded_layout_t::multi_index_t;
    using container_t          = containers::static_tensor<value_type, padded_layout_t>;

private:
    static constexpr auto s_rank       = patch_layout_t::rank();
    static constexpr auto s_halo_width = patch_layout_t::halo_width();

public:
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
    constexpr auto data() noexcept -> container_t&
    {
        return m_data;
    }

    [[nodiscard]]
    constexpr auto data() const noexcept -> container_t const&
    {
        return m_data;
    }

private:
    template <concepts::Direction auto Direction>
    [[nodiscard]]
    static constexpr auto halo_iteration_control_impl() -> auto
    {
        using direction_t           = std::remove_cvref_t<decltype(Direction)>;
        static constexpr auto h     = halo_width();
        static constexpr auto rank  = direction_t::rank();
        static constexpr auto sizes = padded_layout_t::sizes();
        using vec_t                 = containers::static_vector<index_t, rank>;

        constexpr auto low       = direction_t::is_negative(Direction);
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
        return containers::manipulators::
            loop_control<typename padded_layout_t::shape_t, start, end, index_t{ 1 }>{};
    }

public:
    template <concepts::Direction auto Direction>
    using halo_iteration_control_t = decltype(halo_iteration_control_impl<Direction>());

    [[nodiscard]]
    static constexpr auto from_container(container_t const& c) noexcept -> patch
    {
        patch p{};
        p.m_data = c;
        return p;
    }

public:
    template <class... I>
        requires(sizeof...(I) == s_rank) && (std::integral<std::remove_cvref_t<I>> && ...)
    [[nodiscard]]
    constexpr auto operator[](I&&... idxs) const noexcept -> const_reference
    {
        return m_data[std::forward<decltype(idxs)>(idxs)...];
    }

    template <class... I>
        requires(sizeof...(I) == s_rank) && (std::integral<std::remove_cvref_t<I>> && ...)
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
