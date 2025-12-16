#ifndef AMR_INCLUDED_NDT_STRUCTURED_PRINT
#define AMR_INCLUDED_NDT_STRUCTURED_PRINT

#include "containers/container_manipulations.hpp"
#include <algorithm>
#include <bitset>
#include <ostream>

namespace amr::ndt::print
{

struct structured_print
{
public:
    structured_print(std::ostream& os) noexcept
        : m_os(os)
    {
    }

    auto print(auto const& tree) const -> void
    {
        using tree_t        = std::remove_cvref_t<decltype(tree)>;
        using patch_shape_t = typename tree_t::patch_layout_t::data_layout_t::shape_t;
        using index_t       = typename patch_shape_t::index_t;
        using tree_index_t  = typename tree_t::patch_index_t;
        using map_t         = typename tree_t::deconstructed_raw_map_types_t;
        using lc_t          = amr::containers::control::
            loop_control<patch_shape_t, index_t{}, patch_shape_t::sizes(), index_t{ 1 }>;
        static constexpr auto rank = patch_shape_t::rank();

        static constexpr auto           size        = tree_index_t::max_depth();
        static constexpr decltype(size) k           = 2;
        static constexpr auto           prefix_pool = [] constexpr -> auto
        {
            std::array<char, size + k> arr{};
            std::ranges::fill(arr, '\t');
            arr[size]     = ' ';
            arr[size + 1] = '{';
            return arr;
        }();
        static constexpr auto prefixes = []
        {
            std::array<std::string_view, size> arr{};
            for (std::size_t d = 0; d != size; ++d)
            {
                arr[d] = std::string_view(prefix_pool.data() + size - d, d + k);
            }
            return arr;
        }();

        for (auto i = 0uz; i != tree.size(); ++i)
        {
            auto const h = tree.get_node_index_at(i);

            print_header(m_os, tree_index_t::level(h))
                << "h: " << std::bitset<tree_index_t::bits()>(h.id()).to_string()
                << ", offset: " << decltype(h)::offset_of(h) << '\n';

            containers::manipulators::shaped_for<lc_t>(
                [this, &tree, &h, i](auto const& idxs)
                {
                    auto prefix = [&h](auto r, auto const& iidxs) constexpr noexcept
                    {
                        // TODO: This was copied from tensor ostream
                        // Write a function that prints a tuple of tensors,
                        // which is what we need here
                        return (r == 0) ? (iidxs[rank - 1] == 0)
                                              ? prefixes[tree_index_t::level(h)]
                                              : ", {"
                                        : "";
                    };

                    static constexpr auto spacer =
                        [](auto r, auto const& iidxs) constexpr noexcept
                    {
                        return (r + 1 == rank) ? (iidxs[rank - 1] + 1 ==
                                                  patch_shape_t::size(rank - 1))
                                                     ? "}\n"
                                                     : "}"
                                               : ", ";
                    };
                    [this, &tree, &h, i, &idxs, &prefix]<std::size_t... I>(
                        std::index_sequence<I...>
                    )
                    {
                        (
                            (m_os
                             << prefix(I, idxs)
                             << tree.template get_patch<std::tuple_element_t<I, map_t>>(i)
                                    .data()[idxs]
                             << spacer(I, idxs)),
                            ...
                        );
                    }(std::make_index_sequence<std::tuple_size_v<map_t>>{});
                }
            );
        }
    }

private:
    static auto print_header(std::ostream& os, auto depth) -> std::ostream&
    {
        for (auto i = decltype(depth){}; i != depth; ++i)
        {
            os << '\t';
        }
        return os;
    }

    std::ostream& m_os;
};

} // namespace amr::ndt::print

#endif // AMR_INCLUDED_NDT_STRUCTURED_PRINT
