#ifndef GLOBALS_HPP
#define GLOBALS_HPP

#include <array>
#include <map>
#include <vector>
#include <tuple>
#include <cassert>

namespace amr::basis {
    template<unsigned int Dimensions, unsigned int Order>
    class Basis;
}

namespace amr::globals {

enum class Face {
    Left,
    Right,
    Back,
    Front,
    Bottom,
    Top,
    
    FaceCount
};

inline const std::array<const char*, 6> face_names = {
    "Left", "Right", "Bottom", "Top", "Back", "Front"
};

inline const char* to_string(Face f) {
    return face_names[static_cast<int>(f)];
}

struct Globals {
    // == Topology and orientation ==
    std::map<Face, int> normal_signs = {
        {Face::Left, -1}, {Face::Right, 1},
        {Face::Bottom, -1}, {Face::Top, 1},
        {Face::Back, -1}, {Face::Front, 1}
    };

    std::map<Face, int> normal_idxs = {
        {Face::Left, 0}, {Face::Right, 0},
        {Face::Bottom, 1}, {Face::Top, 1},
        {Face::Back, 2}, {Face::Front, 2}
    };

    std::map<Face, Face> opposite_faces = {
        {Face::Left, Face::Right},
        {Face::Right, Face::Left},
        {Face::Bottom, Face::Top},
        {Face::Top, Face::Bottom},
        {Face::Back, Face::Front},
        {Face::Front, Face::Back}
    };

    // == Precomputed projections ==
    std::map<Face, std::vector<std::vector<double>>> project_dofs_to_face;
    std::map<Face, std::vector<std::vector<double>>> project_flux_to_face;
    std::map<Face, std::vector<std::vector<double>>> project_dofs_from_face;

    // == Quadrature and matrix data ==
    std::vector<double> quadweights_nd;
    std::vector<std::tuple<double, double>> quadpoints_nd;

    std::vector<std::vector<double>> reference_massmatrix_face;
    std::vector<std::vector<double>> reference_massmatrix_cell;
    std::vector<std::vector<double>> reference_derivative_matrix;
    std::vector<std::vector<double>> filter_matrix;


};

} // namespace amr::globals

#endif // GLOBALS_HPP
