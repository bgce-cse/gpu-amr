#include "morton/morton2d.hpp"
#include <iomanip>
#include <iostream>
#include <vector>
// just for playing around with some stuff... nothing serious here
class MortonTree
{
private:
    std::vector<uint64_t> leaf_container;
    std::vector<uint64_t> refinement_container;
    int                   _max_depth;
    int                   _length_x;
    int                   _length_y;

public:
    MortonTree(int max_depth, int length_x, int length_y)
        : _max_depth(max_depth)
        , _length_x(length_x)
        , _length_y(length_y)
    {
        init_construct(1);
    };

    void init_construct(uint8_t initial_refinement)
    {
        // construct tree uniformly as a starting point
        // we start with 4^initial refinement initial cells.
        int delta_x_y =
            1 << (_max_depth - initial_refinement); // 2^(_max_depth - initial_refinement)

        for (int i = 0; i < (1 << initial_refinement); i++) // 2^initial_refinement
        {
            for (int j = 0; j < (1 << initial_refinement); j++) // 2^initial_refinement
            {
                uint64_t morton_id =
                    morton2d::encode2D(i * delta_x_y, j * delta_x_y, initial_refinement);
                leaf_container.push_back(morton_id);
            }
        }
        refinement_container = std::vector<uint64_t>(leaf_container.size(), 0);
        std::cout << "done with construction" << std::endl;
        for (const auto& elem : leaf_container)
        {
            std::cout << elem << " ";
        }
        std::cout << std::endl;
    }

    void reconstruct()
    {
        for (size_t i = 0; i < leaf_container.size(); i++)
        {
            if (refinement_container[i] == 1)
            {
                std::cout << "refining "
                          << std::get<0>(morton2d::decode2D(leaf_container[i])) << " , "
                          << std::get<0>(morton2d::decode2D(leaf_container[i]))
                          << " level : " << std::get<2>(morton2d::decode2D(leaf_container[i])) << std::endl;
                
                auto children   = morton2d::getChild2D(leaf_container[i]);
                auto child1     = std::get<0>(children);
                auto child2     = std::get<1>(children);
                auto child3     = std::get<2>(children);
                auto child4     = std::get<3>(children);
                auto child1_dec = morton2d::decode2D(child1);
                auto child2_dec = morton2d::decode2D(child2);
                auto child3_dec = morton2d::decode2D(child3);
                auto child4_dec = morton2d::decode2D(child4);
                std::cout << "Child 1 coordinates: (" << std::get<0>(child1_dec) << ", "
                          << std::get<1>(child1_dec)
                          << "), level: " << static_cast<int>(std::get<2>(child1_dec))
                          << std::endl;
                std::cout << "Child 2 coordinates: (" << std::get<0>(child2_dec) << ", "
                          << std::get<1>(child2_dec)
                          << "), level: " << static_cast<int>(std::get<2>(child2_dec))
                          << std::endl;
                std::cout << "Child 3 coordinates: (" << std::get<0>(child3_dec) << ", "
                          << std::get<1>(child3_dec)
                          << "), level: " << static_cast<int>(std::get<2>(child3_dec))
                          << std::endl;
                std::cout << "Child 4 coordinates: (" << std::get<0>(child4_dec) << ", "
                          << std::get<1>(child4_dec)
                          << "), level: " << static_cast<int>(std::get<2>(child4_dec))
                          << std::endl;

                leaf_container.push_back(child1);
                leaf_container.push_back(child2);
                leaf_container.push_back(child3);
                leaf_container.push_back(child4);
                refinement_container.push_back(0);
                refinement_container.push_back(0);
                refinement_container.push_back(0);
                refinement_container.push_back(0);
                leaf_container.erase(leaf_container.begin() + i);
                refinement_container.erase(refinement_container.begin() + i);
            }
        }
    }

    void flag_refinement()
    {
        auto refinement_condition = [this](uint64_t morton_id)
        {
            auto     decoded_id = morton2d::decode2D(morton_id);
            uint32_t x_ccord    = std::get<0>(decoded_id);
            uint32_t y_ccord    = std::get<1>(decoded_id);
            uint8_t  level      = std::get<2>(decoded_id);
            if (x_ccord < 0.5 * (1 << _max_depth) && y_ccord < 0.5 * (1 << _max_depth) &&
                level < 2)
            {
                return 1;
            }
            return 0;
        };
        for (size_t i = 0; i < leaf_container.size(); i++)
        {
            if (refinement_condition(leaf_container[i]))
            {
                refinement_container[i] = 1;
            }
        }
    }

    // Add this public method to your MortonTree class
    void print_tree()
    {
        std::cout << "Morton Tree Visualization:\n";
        std::cout << "Max depth: " << _max_depth << ", Grid size: " << _length_x << "x"
                  << _length_y << "\n";
        std::cout << "Number of leaves: " << leaf_container.size() << "\n\n";

        // Print detailed cell information
        std::cout << "Cell details:\n";
        for (size_t i = 0; i < leaf_container.size(); i++)
        {
            auto     decoded = morton2d::decode2D(leaf_container[i]);
            uint32_t x       = std::get<0>(decoded);
            uint32_t y       = std::get<1>(decoded);
            uint8_t  level   = std::get<2>(decoded);
            int      size    = 1 << (_max_depth - level);

            std::cout << "Cell " << i << ": Morton=" << leaf_container[i] << " Pos=(" << x
                      << "," << y << ") Level=" << (int)level << " Size=" << size << "x"
                      << size;

            if (i < refinement_container.size() && refinement_container[i] == 1)
            {
                std::cout << " [FLAGGED]";
            }
            std::cout << "\n";
        }
    }
};