#ifndef AMR_INCLUDED_NDTREE
#define AMR_INCLUDED_NDTREE

#include "ndconcepts.hpp"
#include "ndhierarchy.hpp"
#include "utility/compile_time_utility.hpp"
#include "utility/constexpr_functions.hpp"
#include <algorithm>
#include <concepts>
#include <cstdint>
#include <cstdlib>
#include <flat_map>
#include <tuple>
#include <type_traits>

#ifndef NDEBUG
#    define AMR_NDTREE_CHECKBOUNDS
#endif

namespace amr::ndt::tree
{

template <concepts::DeconstructibleType T, concepts::NodeIndex Node_Index>
class flat_ndtree
{
public:
    using value_type                  = T;
    using node_index_t                = Node_Index;
    using size_type                   = std::size_t;
    using flat_index_t                = size_type;
    using node_index_directon_t       = typename node_index_t::direction_t;
    static constexpr auto s_nd_fanout = node_index_t::nd_fanout();
    template <typename Type>
    using pointer_t = Type*;
    template <typename Type>
    using const_pointer_t = Type const*;

    static_assert(s_nd_fanout > 1);

    template <typename>
    struct deconstructed_buffers_impl;

    template <typename... Ts>
        requires concepts::detail::type_map_tuple_impl<std::tuple<Ts...>>
    struct deconstructed_buffers_impl<std::tuple<Ts...>>
    {
        using type = std::tuple<typename Ts::type*...>;
    };

    using deconstructed_buffers_t =
        typename deconstructed_buffers_impl<typename T::deconstructed_types_map_t>::type;

    template <typename>
    struct deconstructed_types_impl;

    template <typename... Ts>
        requires concepts::detail::type_map_tuple_impl<std::tuple<Ts...>>
    struct deconstructed_types_impl<std::tuple<Ts...>>
    {
        using type = std::tuple<typename Ts::type...>;
    };

    using deconstruced_types_t =
        typename deconstructed_types_impl<typename T::deconstructed_types_map_t>::type;

    using index_map_t                = std::flat_map<node_index_t, flat_index_t>;
    using index_map_iterator_t       = typename index_map_t::iterator;
    using index_map_const_iterator_t = typename index_map_t::const_iterator;

public:
    flat_ndtree(size_type size) noexcept
    {
        std::apply(
            [size](auto&... e)
            {
                ((e = (pointer_t<std::remove_pointer_t<std::remove_cvref_t<decltype(e)>>>)
                      std::malloc(
                          size *
                          sizeof(std::remove_pointer_t<std::remove_cvref_t<decltype(e)>>)
                      )),
                 ...);
            },
            m_data_buffers
        );
    }

    ~flat_ndtree() noexcept
    {
        std::apply([](auto&... e) { (std::free(e), ...); }, m_data_buffers);
    }

public:
    template <concepts::TypeMap Map_Type>
    [[gnu::always_inline, gnu::flatten]]
    auto get() noexcept -> pointer_t<typename Map_Type::type>
    {
        return std::get<pointer_t<typename Map_Type::type>>(m_data_buffers);
    }

    template <concepts::TypeMap Map_Type>
    [[gnu::always_inline, gnu::flatten]]
    auto get() const noexcept -> const_pointer_t<typename Map_Type::type>
    {
        return std::get<pointer_t<typename Map_Type::type>>(m_data_buffers);
    }

public:
    deconstructed_buffers_t m_data_buffers;
    index_map_t             m_index_map;
};

} // namespace amr::ndt::tree

#endif // AMR_INCLUDED_NDTREE
