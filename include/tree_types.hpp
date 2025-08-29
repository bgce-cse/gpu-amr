#pragma once
#include "data_types.hpp"
#include "morton_id.hpp"
#include "tree.hpp"

using index_t = morton::morton_id<10u, 2u>;
template <typename T>
using ndtree_t = ndtree<T, cell, index_t>;
using direction_t = index_t::direction_t;
