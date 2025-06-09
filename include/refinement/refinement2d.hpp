#ifndef AMR_INCLUDED_REFINEMENT
#define AMR_INCLUDED_REFINEMENT

// ===============================
// ! WORK IN PROGRESS !
// This file contains incomplete or experimental code.
// It is not finalized and is subject to change.
// ===============================

#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

using MortonID = uint64_t;

constexpr uint8_t MAX_LEVEL         = 64;
constexpr uint8_t REFINE_THRESHOLD  = 20;
constexpr uint8_t COARSEN_THRESHOLD = 5;

namespace refinement2d
{

// @brief Function to check if a grid point should be refined based on the gradient
// magnitude
/// @param grid 2D vector representing the grid
/// @param id Morton ID of the grid point
/// @return True if the grid point should be refined, false otherwise
bool shouldRefine(const Grid& grid, MortonID id)
{
    auto it = grid.find(id);
    if (it == grid.end() || it->second.level >= MAX_LEVEL) return false;

    double grad = computeGradientMagnitude(grid, id);
    return grad > REFINE_THRESHOLD;
}

// TODO: Implement findCoarsenableGroups() or similar would make more sense?

// @brief Function to check if a grid point should be coarsened based on the gradient
// magnitude
/// @param grid 2D vector representing the grid
/// @param id Morton ID of the grid point
/// @return True if the grid point should be coarsened, false otherwise
bool shouldCoarsen(const Grid& grid, MortonID id)
{
    auto it = grid.find(id);
    if (it == grid.end() || it->second.level <= 0) return false;

    double grad = computeGradientMagnitude(grid, id);
    return grad < COARSEN_THRESHOLD;
}

// @brief Function to compute the gradient magnitude at a grid point
/// @param grid 2D vector representing the grid
/// @param id Morton ID of the grid point
/// @return Gradient magnitude at the specified grid point
double computeGradientMagnitude(const Grid& grid, MortonID id)
{
    auto [i, j, lvl] = morton::decode(id);
    double dx        = 1.0 / (1 << lvl);

    double drho_dx = 0, drho_dy = 0;

    auto neighbors = morton::getNeighbors(id);
    if (neighbors.count("left") && neighbors.count("right"))
    {
        drho_dx =
            (grid.at(neighbors["right"]).rho - grid.at(neighbors["left"]).rho) / (2 * dx);
    }
    if (neighbors.count("top") && neighbors.count("bottom"))
    {
        drho_dy =
            (grid.at(neighbors["top"]).rho - grid.at(neighbors["bottom"]).rho) / (2 * dx);
    }

    return std::sqrt(drho_dx * drho_dx + drho_dy * drho_dy);
}

// @brief Function to generate four child cells for a given parent cell
/// @param parent Parent cell to generate children from
/// @param parent_id Morton ID of the parent cell
/// @return Vector of Morton IDs for the child cells
std::vector<MortonID> generateChildren(const Cell& parent, const MortonID parent_id)
{
    auto [i, j, level] = morton::decode(parent_id);
    int childLevel     = level + 1;
    assert(childLevel < MAX_LEVEL && "refinement level exceeds MAX_LEVEL");

    std::vector<MortonID> children;
    // TODO: children anchor positions matching?
    // Current assumption: i, j indicate cell position, so in a finer level it would be
    // 2*i, 2*j (for bottom-left) Other possibility would be that i, j indicate
    // coordinates, so in a finer level it would be still i and j (for bottom-left)
    children.push_back(morton::encode(2 * i, 2 * j, childLevel));         // bottom-left
    children.push_back(morton::encode(2 * i + 1, 2 * j, childLevel));     // bottom-right
    children.push_back(morton::encode(2 * i, 2 * j + 1, childLevel));     // top-left
    children.push_back(morton::encode(2 * i + 1, 2 * j + 1, childLevel)); // top-right
    return children;
}

// @brief Function to interpolate values from a parent cell to its children
/// @param parent_id Morton ID of the parent cell
/// @param parent Parent cell to interpolate from
/// @param grid 2D vector representing the grid
/// @param target_grid 2D vector representing the target grid where children will be
/// placed
/// @return
void interpolateValues(
    const MortonID parent_id,
    const Cell&    parent,
    const Grid&    grid,
    Grid&          target_grid
)
{
    auto children    = generateChildren(parent, parent_id);
    auto [i, j, lvl] = morton::decode(parent_id);

    // Use parent and neighbors for interpolation
    auto   neighbors = morton::getNeighbors(parent_id);
    double rho_c     = parent.rho;
    double rho_l = neighbors.count("left") ? grid.at(neighbors.at("left")).rho : rho_c;
    double rho_r = neighbors.count("right") ? grid.at(neighbors.at("right")).rho : rho_c;
    double rho_t = neighbors.count("top") ? grid.at(neighbors.at("top")).rho : rho_c;
    double rho_b =
        neighbors.count("bottom") ? grid.at(neighbors.at("bottom")).rho : rho_c;

    // Basic bilinear interpolation (center + edges)
    std::vector<double> rho_interp = {
        (4 * rho_c + 2 * rho_l + 2 * rho_b) / 8.0, // bottom-left
        (4 * rho_c + 2 * rho_r + 2 * rho_b) / 8.0, // bottom-right
        (4 * rho_c + 2 * rho_l + 2 * rho_t) / 8.0, // top-left
        (4 * rho_c + 2 * rho_r + 2 * rho_t) / 8.0  // top-right
    };

    for (int k = 0; k < 4; ++k)
    {
        target_grid[children[k]] = { .rho    = rho_interp[k],
                                     .energy = parent.energy, // Copy scalar values
                                     .level  = parent.level + 1 };
    }
}

// @brief Function to refine cells in the grid
// This function iterates through the grid and refines cells that meet the refinement
// criteria by generating their children and interpolating values from the parent cell.
/// @param grid 2D vector representing the grid
/// @return
void applyRefinement(Grid& grid)
{
    std::vector<MortonID> toRefine;

    // First pass: decide which cells to refine
    // TODO: Only iterate through active cells
    for (const auto& [id, cell] : grid)
    {
        if (shouldRefine(grid, id)) toRefine.push_back(id);
    }

    for (const MortonID& id : toRefine)
    {
        const Cell& parent = grid.at(id);

        // Generate children and interpolate
        interpolateValues(id, parent, grid, grid);

        // Optionally: remove parent (or mark as inactive)
        // grid.erase(id);
    }
}

// @brief Function to coarsen cells in the grid
// This function iterates through the grid and coarsens cells that meet the coarsening
// criteria by averaging values from their children and removing the child cells.
/// @param grid 2D vector representing the grid
/// @return
void applyCoarsening(Grid& grid)
{
    // TODO: Implement coarsening logic
}

} // namespace refinement2d

#endif // AMR_INCLUDED_REFINEMENT