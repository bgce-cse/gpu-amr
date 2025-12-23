#ifndef AMR_INCLUDED_CONTAINER_UTILS
#define AMR_INCLUDED_CONTAINER_UTILS

#include "container_concepts.hpp"
#include "static_layout.hpp"
#include "static_shape.hpp"
#include "static_tensor.hpp"
#include <algorithm>
#include <cassert>
#include <concepts>
#include <type_traits>
#include <utility>

namespace amr::containers::utils
{

namespace types
{

namespace sequences
{

namespace detail
{

template <typename, typename>
struct concatenate_t_impl;

template <std::integral I, I... As, I... Bs>
struct concatenate_t_impl<
    std::integer_sequence<I, As...>,
    std::integer_sequence<I, Bs...>>
{
    using type = std::integer_sequence<I, As..., Bs...>;
};

template <concepts::Container auto S1, concepts::Container auto S2>
struct concatenate_v_impl
{
    static constexpr auto value = []
    {
        using size_type = std::common_type_t<
            typename decltype(S1)::value_type,
            typename decltype(S2)::value_type>;
        constexpr auto           N = std::ranges::size(S1) + std::ranges::size(S2);
        std::array<size_type, N> ret{};
        auto next = std::copy(std::cbegin(S1), std::cend(S1), std::begin(ret));
        std::copy(std::cbegin(S2), std::cend(S2), next);
        std::ranges::copy(S2, next);
        return ret;
    }();
};

} // namespace detail

template <typename A, typename B>
using concatenate_t = typename detail::concatenate_t_impl<A, B>::type;

template <auto A, auto B>
static constexpr auto concatenate_v = detail::concatenate_v_impl<A, B>::value;

} // namespace sequences

namespace shape
{

template <typename>
struct static_shape_wrapper;

template <std::integral I, I... Ns>
struct static_shape_wrapper<std::integer_sequence<I, Ns...>>
{
    using type = static_shape<Ns...>;
};

template <typename T>
using static_shape_wrapper_t = typename static_shape_wrapper<T>::type;

} // namespace shape

namespace layout
{

template <concepts::StaticLayout Layout>
struct padded_layout
{
    using size_type = typename Layout::size_type;

    static constexpr auto padded_size = []<std::integral auto Pad>
    {
        static constexpr auto pad = static_cast<size_type>(Pad);
        static_assert(pad > size_type{});
        auto sizes = Layout::sizes();
        for (auto& e : sizes)
            e += pad;
        return sizes;
    };

    template <std::integral auto Pad>
    using type = static_layout<static_shape<padded_size.template operator()<Pad>()>>;
};

} // namespace layout

namespace tensor
{

namespace detail
{

template <typename T, std::integral auto Size, std::size_t Rank, std::size_t... Is>
constexpr auto make_hypercube_type_impl(std::index_sequence<Is...>)
    -> static_tensor<T, static_layout<static_shape<std::array{ ((void)Is, Size)... }>>>
{
    return {};
}

} // namespace detail

template <typename T, std::integral auto Size, std::size_t Rank>
struct hypercube
{
    using type = decltype(detail::make_hypercube_type_impl<T, Size, Rank>(
        std::make_index_sequence<Rank>{}
    ));
};

template <typename T, std::integral auto Size, std::size_t Rank>
using hypercube_t = typename hypercube<T, Size, Rank>::type;

template <concepts::StaticContainer A, concepts::StaticContainer B>
struct tensor_product_result
{
    using a_vt       = typename A::value_type;
    using b_vt       = typename B::value_type;
    using value_type = typename std::conditional_t<
        std::is_scalar_v<a_vt> && std::is_scalar_v<b_vt>,
        std::common_type_t<a_vt, b_vt>,
        std::conditional_t<
            std::is_scalar_v<a_vt>,
            std::type_identity_t<b_vt>,
            std::type_identity_t<a_vt>>>;
    static constexpr auto result_size = sequences::concatenate_v<A::sizes(), B::sizes()>;
    using type = static_tensor<value_type, static_layout<static_shape<result_size>>>;
};

template <concepts::StaticContainer A, concepts::StaticContainer B>
using tensor_product_result_t = typename tensor_product_result<A, B>::type;

template <std::integral Index_Type, std::integral auto Order>
struct contraction_index_set
{
public:
    using index_t                 = Index_Type;
    using size_type               = index_t;
    static constexpr auto s_order = Order;
    using index_pair_t            = std::pair<index_t, index_t>;
    using container_t             = std::array<index_pair_t, s_order>;
    using const_iterator          = typename container_t::const_iterator;
    using iterator                = typename container_t::iterator;
    using value_type              = typename container_t::value_type;

    static_assert(s_order >= 0);

public:
    explicit constexpr contraction_index_set(
        std::same_as<index_pair_t> auto const&... index_pairs
    )
        requires(sizeof...(index_pairs) == s_order)
        : indices_{ index_pairs... }
    {
    }

    [[nodiscard]]
    static constexpr auto order() noexcept -> auto
    {
        return s_order;
    }

    [[nodiscard]]
    constexpr auto operator[](index_t const i) const noexcept -> index_pair_t const&
    {
        assert(i < s_order);
        if constexpr (std::is_signed_v<index_t>)
        {
            assert(i >= index_t{});
        }
        return indices_[i];
    }

    [[nodiscard]]
    constexpr auto cbegin() const noexcept -> const_iterator
    {
        return std::cbegin(indices_);
    }

    [[nodiscard]]
    constexpr auto cend() const noexcept -> const_iterator
    {
        return std::cend(indices_);
    }

    [[nodiscard]]
    constexpr auto begin() const noexcept -> const_iterator
    {
        return std::begin(indices_);
    }

    [[nodiscard]]
    constexpr auto end() const noexcept -> const_iterator
    {
        return std::end(indices_);
    }

    [[nodiscard]]
    constexpr auto begin() noexcept -> iterator
    {
        return std::begin(indices_);
    }

    [[nodiscard]]
    constexpr auto end() noexcept -> iterator
    {
        return std::end(indices_);
    }

public:
    container_t indices_;
};

template <
    concepts::StaticContainer          A,
    concepts::StaticContainer          B,
    concepts::ContractionIndexSet auto CIS>
struct tensor_contraction_result
{
    using size_type  = std::common_type_t<typename A::size_type, typename B::size_type>;
    using value_type = std::common_type_t<typename A::value_type, typename B::value_type>;

    static constexpr auto s_out_sizes = []
    {
        constexpr size_type s_in_rank  = A::rank() + B::rank();
        constexpr size_type s_out_rank = s_in_rank - 2 * CIS.order();
        using ret_t                    = std::array<size_type, s_out_rank>;
        // constexpr auto sizes           = sequences::concatenate_v<A::sizes(), B::sizes()>;
        ret_t          ret{};

        size_type k = 0;
        for (auto i = size_type{}; i != A::rank(); ++i)
        {
            if (std::ranges::find_if(
                    CIS, [&i](auto const& p) { return p.first == i; }
                ) != std::cend(CIS))
            {
                continue;
            }
            ret[k++] = A::size(i);
        }
        for (auto j = size_type{}; j != B::rank(); ++j)
        {
            if (std::ranges::find_if(
                    CIS, [&j](auto const& p) { return p.second == j; }
                ) != std::cend(CIS))
            {
                continue;
            }
            ret[k++] = B::size(j);
        }
        assert(k == s_out_rank);

        return ret;
    }();

    using type = static_tensor<value_type, static_layout<static_shape<s_out_sizes>>>;
};

template <
    concepts::StaticContainer          A,
    concepts::StaticContainer          B,
    concepts::ContractionIndexSet auto CIS>
using tensor_contraction_result_t = typename tensor_contraction_result<A, B, CIS>::type;

} // namespace tensor

} // namespace types

} // namespace amr::containers::utils

#endif // AMR_INCLUDED_CONTAINER_UTILS
