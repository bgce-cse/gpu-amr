# Tree Direction System Explanation

## Overview
The AMR tree uses a **natural direction ordering** system where directions are systematically numbered to iterate through neighbors in a predictable pattern. The direction dimensions directly correspond to the layout dimensions because the layout uses **row-major (C-style) ordering**.

## Direction Natural Ordering (2D Case)

For a 2D domain, there are **4 neighbors** per cell. The patch layout is `[dim0][dim1]` where dim1 varies fastest.

### Direction Dimension Mapping

**The key insight:** Direction dimensions 0 and 1 directly correspond to layout dimensions 0 and 1:

```
Direction Dimension 0  ↔  Layout Dimension 0  ↔  NORTH/SOUTH (Y-axis in typical visual)
Direction Dimension 1  ↔  Layout Dimension 1  ↔  EAST/WEST (X-axis in typical visual)
```

### Direction Index Mapping
```
d.index() = 0  →  Dimension 0, NEGATIVE direction  (SOUTH, -Y)
d.index() = 1  →  Dimension 0, POSITIVE direction  (NORTH, +Y)
d.index() = 2  →  Dimension 1, NEGATIVE direction  (WEST, -X)
d.index() = 3  →  Dimension 1, POSITIVE direction  (EAST, +X)
```

### Key Formulas
```cpp
d.dimension()  = d.index() / 2              // Which dimension (0 or 1)
is_negative()  = (d.index() % 2 == 0)       // true for indices 0, 2 (SOUTH/WEST)
is_positive()  = (d.index() % 2 == 1)       // true for indices 1, 3 (NORTH/EAST)
```

## Why Dimension 0 Gives North/South Neighbors

The layout strides (from `static_layout.hpp`) determine memory traversal:

```cpp
// For 2D shape [size_0, size_1]:
strides[0] = size_1;  // Last dimension's size
strides[1] = 1;       // Last dimension has stride 1
```

This is **row-major order**: the last index varies fastest in memory. Therefore:
- **Dimension 0** is the "outer" loop → changing it jumps by `size_1` positions → **moves North/South**
- **Dimension 1** is the "inner" loop → changing it jumps by 1 position → **moves East/West**

### Visual Example
```
Layout: [dim0][dim1]  where dim1 varies fastest

Memory layout (for 4x4 grid):
Position  Index  Coords[0,1]
0         0      [0,0]
1         1      [0,1]
2         2      [0,2]
3         3      [0,3]
4         4      [1,0]  ← changing dim0 from [0,*] to [1,*]
...

To move North (increase dim0):  old_pos=3 → new_pos=7 (jump by size_1=4)
To move East  (increase dim1):  old_pos=3 → new_pos=4 (jump by 1)
```

## Structure Definition

```cpp
// From include/ndtree/neighbor.hpp
template <std::integral auto Dim>
struct direction
{
private:
    static constexpr auto s_neighbors_per_dim = 2;           // 2 per dimension
    static constexpr auto s_elements = s_neighbors_per_dim * Dim;  // Total directions
    static constexpr auto s_collection = {0, 1, 2, 3};       // For 2D
    
    index_t idx_;  // Current direction index (0-3)
    
public:
    [[nodiscard]]
    static constexpr auto is_negative(direction d) -> bool
    {
        return d.index() % 2 == 0;  // Returns true for 0, 2
    }
    
    [[nodiscard]]
    constexpr auto dimension() -> index_t
    {
        return index() / 2;  // Returns 0 or 1
    }
};
```

## Iteration Pattern

When iterating through directions:
```cpp
for (auto d = direction_t::first(); d != direction_t::sentinel(); d.advance())
{
    // d.index() will take values: 0 → 1 → 2 → 3 → sentinel
}
```

**Iteration sequence:**
1. **First iteration** (d.index() = 0): SOUTH neighbor (dimension 0, negative)
2. **Second iteration** (d.index() = 1): NORTH neighbor (dimension 0, positive)
3. **Third iteration** (d.index() = 2): WEST neighbor (dimension 1, negative)
4. **Fourth iteration** (d.index() = 3): EAST neighbor (dimension 1, positive)

## Memory Layout: Row-Major Order

Your patches use `static_layout<shape_t>` which implements **C-style row-major storage**:
- Layout is stored as `[dim0][dim1]`
- Dimension 1 (the rightmost subscript) varies fastest in memory
- Linear index = coord[0] * size[1] + coord[1]

**No dimension swapping needed** - the direction dimensions map directly to layout dimensions!

## Correct Coordinate Computation

```cpp
// Given direction d with dim_in_direction and is_negative_dir
const std::size_t layout_dim = dim_in_direction;  // Direct mapping, NO SWAP!
auto neighbor_coords = current_coords;
neighbor_coords[layout_dim] += (is_negative_dir ? -1 : 1);
```

### Example: Cell at coords (2, 3), want South neighbor (direction 0)
```
current_coords = [2, 3]
dim_in_direction = 0 (south/north direction)
is_negative_dir = true (south = negative)
layout_dim = 0  (directly from dim_in_direction)

neighbor_coords[0] += -1  →  neighbor_coords = [1, 3]
```

This moves from row 2 to row 1 (southward) ✓

## Summary Table

| d.index() | dimension() | is_negative() | Direction Name  | Coordinate Change | Semantic |
|-----------|------------|---------------|-----------------|-------------------|----------|
| 0         | 0          | true          | SOUTH (-Y)      | coords[0] -= 1    | Up in grid |
| 1         | 0          | false         | NORTH (+Y)      | coords[0] += 1    | Down in grid |
| 2         | 1          | true          | WEST (-X)       | coords[1] -= 1    | Left in grid |
| 3         | 1          | false         | EAST (+X)       | coords[1] += 1    | Right in grid |



