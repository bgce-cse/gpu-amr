#include "dg_helpers/basis.hpp"
#include "dg_helpers/equations/advection.hpp"
#include "dg_helpers/equations/euler.hpp"
#include "dg_helpers/surface.hpp"
#include "generated_config.hpp"
#include <iomanip>
#include <iostream>
#include <memory>

using namespace amr::equations;
using namespace amr::Basis;
using namespace amr::containers;
using namespace amr::config;

void test_advection_2d()
{
    std::cout << "\n========== Test 2D Advection ==========" << std::endl;

    auto adv =
        std::make_unique<Advection2D<DOFs, Order, Dim, double>>(1.0); // velocity = 1.0

    // Test position
    static_vector<double, Dim> pos{ 0.2, 0 };
    double                     t = 0.0;

    // Get initial condition
    auto dofs = adv->get_2D_initial_values(pos, t);

    std::cout << "Position: (" << pos[0] << ", " << pos[1] << ")\n";
    std::cout << "Time: t = " << t << "\n";
    std::cout << "DOF values (components): [";
    for (unsigned int i = 0; i < DOFs; ++i)
    {
        std::cout << std::setprecision(6) << dofs[i];
        if (i < DOFs - 1) std::cout << ", ";
    }
    std::cout << "]\n";

    // Analytical solution for advection initial conditions
    const double PI = 3.14159265358979323846;
    const double x  = pos[0];
    const double y  = pos[1];

    double analytical_u0 = std::sin(2.0 * PI * (x + y - 2.0 * t));
    double analytical_u1 = std::sin(2.0 * PI * (y - t));

    std::cout << "\nAnalytical Solution Comparison:\n";
    std::cout << "  U[0] computed:   " << std::setprecision(6) << dofs[0] << "\n";
    std::cout << "  U[0] analytical: " << analytical_u0 << "\n";
    std::cout << "  Error: " << std::abs(dofs[0] - analytical_u0) << "\n";

    std::cout << "  U[1] computed:   " << dofs[1] << "\n";
    std::cout << "  U[1] analytical: " << analytical_u1 << "\n";
    std::cout << "  Error: " << std::abs(dofs[1] - analytical_u1) << "\n";

    std::cout << "\nVelocity: " << adv->get_velocity() << "\n";

    // Note: max_eigenvalue expects a dof_t (tensor), not dof_value_t (vector)
    // get_2D_initial_values returns dof_value_t (the values at a point)
    // For eigenvalue testing, we would need a full cell's DOF tensor

    std::cout << "✓ Advection 2D test complete\n";
}

void test_euler_2d()
{
    std::cout << "\n========== Test 2D Euler ==========" << std::endl;

    auto euler = std::make_unique<Euler<Order, Dim, double>>(1.4); // gamma = 1.4 for air

    // Test position
    static_vector<double, Dim> pos{ 0.5, 0.5 };
    double                     t = 0.0;

    // Get initial condition (Gaussian pressure bump)
    auto dofs = euler->get_2D_initial_values(pos, t);

    std::cout << "Position: (" << pos[0] << ", " << pos[1] << ")\n";
    std::cout << "Time: t = " << t << "\n";
    std::cout << "Conservative variables [rhou, rhov, rho, E]: [";
    for (unsigned int i = 0; i < 4; ++i)
    {
        std::cout << std::setprecision(6) << dofs[i];
        if (i < 3) std::cout << ", ";
    }
    std::cout << "]\n";

    // Analytical solution for Gaussian wave scenario
    const double x = pos[0];
    const double y = pos[1];

    // Pressure bump centered at (0.5, 0.5)
    double p =
        std::exp(-100.0 * (x - 0.5) * (x - 0.5) - 100.0 * (y - 0.5) * (y - 0.5)) + 1.0;
    double gamma = euler->get_gamma();

    // E = p/(gamma-1) + 0.5*(rhou^2 + rhov^2)/rho
    double E_analytical = p / (gamma - 1.0); // u=v=0, so kinetic energy = 0

    std::cout << "\nAnalytical Solution Comparison:\n";
    std::cout << "  ρu computed:   " << std::setprecision(6) << dofs[0]
              << " (expected: 0.0)\n";
    std::cout << "  ρv computed:   " << dofs[1] << " (expected: 0.0)\n";
    std::cout << "  ρ computed:    " << dofs[2] << " (expected: 1.0)\n";
    std::cout << "  E computed:    " << dofs[3] << "\n";
    std::cout << "  E analytical:  " << E_analytical << "\n";
    std::cout << "  Error in E:    " << std::abs(dofs[3] - E_analytical) << "\n";

    std::cout << "  Pressure at center: " << p << "\n";

    std::cout << "✓ Euler 2D test complete\n";
}

int main()
{
    std::cout << "====================================\n";
    std::cout << "  DG Equation Tests\n";
    std::cout << "====================================\n";

    test_advection_2d();
    test_euler_2d();

    std::cout << "\n====================================\n";
    std::cout << "  All tests completed successfully!\n";
    std::cout << "====================================\n";

    return 0;
}
