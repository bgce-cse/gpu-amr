#include "containers/container_utils.hpp"
#include "containers/static_layout.hpp"
#include "containers/static_tensor.hpp"
#include "ndtree/ndutils.hpp"
#include "ndtree/patch.hpp"
#include "ndtree/patch_layout.hpp"
#include "utility/constexpr_functions.hpp"
#include <functional>
#include <iostream>

int main()
{
    using namespace amr::containers;
    constexpr auto N = 4;
    [[maybe_unused]]
    constexpr auto M    = 6;
    constexpr auto Halo = 1;
    using index_t       = std::int32_t;
    [[maybe_unused]]
    constexpr auto Fanout = 2;
    using shape_t         = amr::containers::static_shape<N, M>;
    using patch_layout_t  = amr::ndt::patches::patch_layout<static_layout<shape_t>, Halo>;
    using patch_t         = amr::ndt::patches::patch<index_t, patch_layout_t>;
    using tensor_t        = typename patch_t::container_t;

    tensor_t from{};
    for (index_t i = 0; auto& e : from)
    {
        e = amr::ndt::utils::patches::is_halo_cell<patch_layout_t>(i) ? -1 : i;
        i++;
    }
    std::cout << "Original indexing:\n";
    std::cout << from << '\n';

    std::cout << "Mapped patches:\n";
    constexpr auto patch_maps =
        amr::ndt::utils::patches::fragmentation_patch_maps<patch_layout_t, Fanout>();
    for (int i = 0; auto const& p : patch_maps)
    {
        std::cout << "Patch " << i++ << '\n' << p << '\n';
    }
}
