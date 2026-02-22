#ifndef GLOBAL_HPP
#define GLOBAL_HPP

#include "amr_indicators.hpp"
#include "coordinates.hpp"
#include "kernels.hpp"
#include "quadrature.hpp"

namespace amr::global
{
using ::amr::global::area;
using ::amr::global::compute_cell_center;
using ::amr::global::compute_patch_corner;
using ::amr::global::global_to_reference;
using ::amr::global::linear_to_local_coords;
using ::amr::global::reference_to_global;
using ::amr::global::remove_halo;
using ::amr::global::volume;

} // namespace amr::global

#endif // GLOBAL_HPP
