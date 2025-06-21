#include "ndtree/morton_tree.hpp"

int main() {
    morton2d::initialize(2);
    MortonTree morton_tree(2, 1, 1);
    morton_tree.print_tree();
    morton_tree.flag_refinement();
    morton_tree.print_tree();
    morton_tree.reconstruct();
    morton_tree.print_tree();

    return 0;
}