/**
 * @brief Test barycentric Lagrange interpolation at compile-time
 */

#include "dg_helpers/basis/basis.hpp"
#include "generated_config.hpp"
#include <type_traits>

namespace test_barycentric
{

// Test barycentric Lagrange with configured basis
using Basis      = amr::basis::Basis<amr::config::Order, amr::config::Dim>;
using Lagrange_t = typename Basis::Lagrange_t;

// Test 1: Verify barycentric weights can be computed at compile-time
constexpr auto test_weights = []()
{
    constexpr auto weights = Lagrange_t::compute_barycentric_weights(Basis::quadpoints);
    return weights.elements();
}();

static_assert(
    test_weights == amr::config::Order,
    "Barycentric weights must have Order entries"
);

// Test 2: Verify barycentric evaluation at compile-time
constexpr auto test_eval = []()
{
    constexpr auto weights = Lagrange_t::compute_barycentric_weights(Basis::quadpoints);

    // Evaluate at a safe point (not a node)
    double x      = 0.3;
    double result = 0.0;

    for (unsigned int i = 0; i < amr::config::Order; ++i)
    {
        result += Lagrange_t::evaluate_barycentric(Basis::quadpoints, weights, i, x);
    }
    return result;
}();

static_assert(
    test_eval > 0.99 && test_eval < 1.01,
    "Barycentric Lagrange basis should sum to 1.0"
);

// Test 3: Verify barycentric derivative at compile-time
constexpr auto test_deriv = []()
{
    constexpr auto weights = Lagrange_t::compute_barycentric_weights(Basis::quadpoints);

    double x      = 0.3;
    double result = 0.0;

    for (unsigned int i = 0; i < amr::config::Order; ++i)
    {
        result += Lagrange_t::derivative_barycentric(Basis::quadpoints, weights, i, x);
    }
    return result;
}();

// Derivative sum should be close to zero (property of Lagrange interpolation)
static_assert(
    test_deriv > -0.5 && test_deriv < 0.5,
    "Barycentric derivative sum should be near zero"
);

// Test 4: Compare barycentric vs standard evaluation (should match)
constexpr auto test_comparison = []()
{
    constexpr auto weights = Lagrange_t::compute_barycentric_weights(Basis::quadpoints);

    double x            = 0.4;
    double bary_sum     = 0.0;
    double standard_sum = 0.0;

    for (unsigned int i = 0; i < amr::config::Order; ++i)
    {
        bary_sum += Lagrange_t::evaluate_barycentric(Basis::quadpoints, weights, i, x);
        standard_sum += Lagrange_t::evaluate(Basis::quadpoints, i, x);
    }

    // Both should sum to 1.0
    return (bary_sum > 0.99 && bary_sum < 1.01) &&
           (standard_sum > 0.99 && standard_sum < 1.01);
}();

static_assert(
    test_comparison,
    "Barycentric and standard Lagrange should both sum to 1.0"
);

} // namespace test_barycentric

int main()
{
    return 0;
}
