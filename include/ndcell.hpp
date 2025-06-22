#ifndef GPU_AMR_NDTREE
#define GPU_AMR_NDTREE

#include "detail/type_traits.hpp"
#include <cstdint>

namespace ndt::cell
{

using dim_t = std::uint32_t;

template <typename Value_Type, dim_t N>
class Cell
{
public:
    using value_type = Value_Type;

private:
};

} // namespace ndt::cell

#endif GPU_AMR_NDTREE
