#ifndef BASIS_HPP
#define BASIS_HPP

// Include all basis components
#include "basis/gauss_legendre.hpp"
#include "basis/lagrange.hpp"
#include "basis/legendre.hpp"
#include "basis/polynomial.hpp"

// Re-export Basis class and GaussLegendre for backward compatibility
namespace amr::Basis
{
using amr::basis::Basis;
using amr::basis::GaussLegendre;
} // namespace amr::Basis

#endif // BASIS_HPP
