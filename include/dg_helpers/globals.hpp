#ifndef GLOBAL_HPP
#define GLOBAL_HPP

/**
 * @brief Global configuration and basis access point
 *
 * All components are fully compile-time available.
 * Access basis through GlobalConfig:
 *   using MyBasis = GlobalConfig<Order,Dim,PatchSize,HaloWidth>::Basis_t;
 */
#include "globals/globals.hpp"

#endif // GLOBAL_HPP
