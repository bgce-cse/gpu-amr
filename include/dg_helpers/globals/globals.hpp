#ifndef GLOBAL_HPP
#define GLOBAL_HPP

// Include all global components in order
#include "coordinates.hpp"
#include "kernels.hpp"
#include "quadrature.hpp"

// Re-export coordinate transformation functions and basis for backward compatibility
namespace amr::global
{
using ::amr::global::area;
using ::amr::global::compute_cell_center;
using ::amr::global::global_to_reference;
using ::amr::global::linear_to_local_coords;
using ::amr::global::reference_to_global;
using ::amr::global::remove_halo;
using ::amr::global::volume;

// Basis is available through GlobalConfig (compile-time access)
// Users should access basis via: GlobalConfig<Order,Dim,PatchSize,HaloWidth>::Basis_t
} // namespace amr::global

#endif // GLOBAL_HPP
