#include "containers/container_utils.hpp"
#include "containers/static_layout.hpp"
#include "containers/static_tensor.hpp"
#include "ndtree/ndutils.hpp"
#include "utility/constexpr_functions.hpp"
#include <functional>
#include <iostream>

int main()
{
    using namespace amr::containers;
    constexpr auto N = 6;
    constexpr auto M = 8;
    using index_t    = std::uint32_t;
    [[maybe_unused]]
    constexpr auto Fanout = 2;
    using tensor_t        = static_tensor<index_t, static_layout<N, M>>;

    tensor_t from{};
    std::ranges::iota(from, 0);
    std::cout << "Original indexing:\n";
    std::cout << from << '\n';

    std::cout << "Mapped patches:\n";
    constexpr auto patch_maps =
        amr::ndt::utils::patches::fragmentation_patch_maps<index_t, Fanout, typename tensor_t::layout_t>(
        );
    int i = 0;
    for (auto const& p : patch_maps)
    {
        std::cout << "Patch " << i++ << '\n' << p << '\n';
    }
}
