#ifndef BASIS_HPP
#define BASIS_HPP

#include "../containers/container_concepts.hpp"
#include "../containers/container_operations.hpp"
#include "../containers/container_utils.hpp"
#include "../containers/static_matrix.hpp"
#include "../containers/static_tensor.hpp"
#include "../containers/static_vector.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <functional>

namespace amr::Basis
{
template <auto N>
using vector = amr::containers::static_vector<double, N>;

/**
 * @brief Pre-computed Gauss-Legendre quadrature data template on [-1, 1]
 */
template <unsigned int Order>
struct QuadData;

template <>
struct QuadData<1>
{
    static constexpr std::array<double, 1> points  = { 0.0 };
    static constexpr std::array<double, 1> weights = { 2.0 };
};

template <>
struct QuadData<2>
{
    static constexpr std::array<double, 2> points  = { -0.5773502691896257,
                                                       0.5773502691896257 };
    static constexpr std::array<double, 2> weights = { 1.0, 1.0 };
};

template <>
struct QuadData<3>
{
    static constexpr std::array<double, 3> points  = { -0.7745966692414834,
                                                       0.0,
                                                       0.7745966692414834 };
    static constexpr std::array<double, 3> weights = { 0.5555555555555556,
                                                       0.8888888888888888,
                                                       0.5555555555555556 };
};

template <>
struct QuadData<4>
{
    static constexpr std::array<double, 4> points  = { -0.8611363115940526,
                                                       -0.3399810435848563,
                                                       0.3399810435848563,
                                                       0.8611363115940526 };
    static constexpr std::array<double, 4> weights = { 0.3478548451374538,
                                                       0.6521451548625461,
                                                       0.6521451548625461,
                                                       0.3478548451374538 };
};

template <>
struct QuadData<5>
{
    static constexpr std::array<double, 5> points  = { -0.9061798459386640,
                                                       -0.5384693101056831,
                                                       0.0,
                                                       0.5384693101056831,
                                                       0.9061798459386640 };
    static constexpr std::array<double, 5> weights = { 0.2369268850561891,
                                                       0.4786286704993665,
                                                       0.5688888888888889,
                                                       0.4786286704993665,
                                                       0.2369268850561891 };
};

template <>
struct QuadData<6>
{
    static constexpr std::array<double, 6> points = {
        -0.9324695142031521, -0.6612093864662645, -0.2386191860831969,
        0.2386191860831969,  0.6612093864662645,  0.9324695142031521
    };
    static constexpr std::array<double, 6> weights = {
        0.1713244923791704, 0.3607615730481386, 0.4679139345726910,
        0.4679139345726910, 0.3607615730481386, 0.1713244923791704
    };
};

template <>
struct QuadData<7>
{
    static constexpr std::array<double, 7> points = {
        -0.9491079123427585, -0.7415311855993945, -0.4058451513773972, 0.0,
        0.4058451513773972,  0.7415311855993945,  0.9491079123427585
    };
    static constexpr std::array<double, 7> weights = {
        0.1294849661688697, 0.2797053914892766, 0.3818300505051189, 0.4179591836734694,
        0.3818300505051189, 0.2797053914892766, 0.1294849661688697
    };
};

template <>
struct QuadData<8>
{
    static constexpr std::array<double, 8> points = {
        -0.9602898564975363, -0.7966664774136267, -0.5255324099163290,
        -0.1834346424956498, 0.1834346424956498,  0.5255324099163290,
        0.7966664774136267,  0.9602898564975363
    };
    static constexpr std::array<double, 8> weights = {
        0.1012285362903763, 0.2223810344533745, 0.3137066458778873, 0.3626837833783620,
        0.3626837833783620, 0.3137066458778873, 0.2223810344533745, 0.1012285362903763
    };
};

/**
 * @brief Class for precomputed 1D Gauss–Legendre quadrature points and weights.
 *
 * @tparam Order Number of quadrature points
 */
template <unsigned int Order>
class GaussLegendre
{
private:
    vector<Order> _quadpoints_1d;
    vector<Order> _quadweights_1d;

public:
    static constexpr unsigned int order = Order;

    /**
     * @brief Construct a Gauss–Legendre quadrature rule on [start, end].
     *
     * Initializes with precomputed nodes on [-1, 1] and maps to [start, end].
     *
     * @param start Left interval bound (default -1)
     * @param end Right interval bound (default 1)
     */
    GaussLegendre(double start = -1.0, double end = 1.0)
    {
        initialize_from_precomputed(start, end);
    }

    /// @return Constant reference to quadrature points
    [[nodiscard]]
    const vector<Order>& points() const
    {
        return _quadpoints_1d;
    }

    /// @return Constant reference to quadrature weights
    [[nodiscard]]
    const vector<Order>& weights() const
    {
        return _quadweights_1d;
    }

private:
    /**
     * @brief Initialize from precomputed data and map to [start, end].
     */
    void initialize_from_precomputed(double start, double end)
    {
        // Copy precomputed values
        constexpr auto& precomputed_points  = QuadData<Order>::points;
        constexpr auto& precomputed_weights = QuadData<Order>::weights;

        for (unsigned int i = 0; i < Order; ++i)
        {
            _quadpoints_1d[i]  = precomputed_points[i];
            _quadweights_1d[i] = precomputed_weights[i];
        }

        // Map from [-1, 1] to [start, end]
        const double half_length = (end - start) / 2.0;
        const double midpoint    = (start + end) / 2.0;

        for (unsigned int i = 0; i < Order; ++i)
        {
            _quadpoints_1d[i] = midpoint + half_length * _quadpoints_1d[i];
            _quadweights_1d[i] *= half_length;
        }
    }
};

/**
 * @brief Compute the Legendre polynomial P_n(x) using the standard recurrence relation.
 *
 * @param n Degree of the polynomial (n >= 0)
 * @param x Evaluation point in [-1, 1]
 * @return Value of the Legendre polynomial P_n(x)
 */
constexpr double legendre(int n, double x)
{
    assert(n >= 0 && "Legendre polynomial degree must be >= 0");

    if (n == 0) return 1.0;
    if (n == 1) return x;

    double Pnm2 = 1.0;
    double Pnm1 = x;
    double Pn   = 0.0;

    for (int k = 2; k <= n; ++k)
    {
        Pn   = ((2.0 * k - 1.0) * x * Pnm1 - (k - 1.0) * Pnm2) / k;
        Pnm2 = Pnm1;
        Pnm1 = Pn;
    }
    return Pn;
}

/**
 * @brief Evaluate the i-th Lagrange interpolation basis polynomial at point x.
 *
 * @tparam N Number of interpolation nodes
 * @param points Vector of nodal points
 * @param i Index of the basis function (0-based)
 * @param x Evaluation point
 * @return Value of L_i(x)
 */
template <auto N>
constexpr double lagrange_1d(const vector<N>& points, std::size_t i, double x)
{
    using size_type = typename std::remove_cvref_t<decltype(points)>::size_type;
    double numerator{ 1.0 };
    double denominator{ 1.0 };
    for (std::size_t k = 0; k < N; ++k)
    {
        if (k == i) continue;
        numerator *= (x - points[static_cast<size_type>(k)]);
        denominator *=
            (points[static_cast<size_type>(i)] - points[static_cast<size_type>(k)]);
    }
    return numerator / denominator;
}

/**
 * @brief Compute the derivative of the j-th 1D Lagrange basis polynomial at x.
 *
 * @tparam N Number of interpolation nodes
 * @param points Vector of nodal points
 * @param j Index of the basis function (0-based)
 * @param x Evaluation point
 * @return Value of L'_j(x)
 */
template <std::size_t N>
constexpr double lagrange_diff(const vector<N>& points, std::size_t j, double x)
{
    if (j >= N) throw std::out_of_range("Index out of range");
    double result = 0.0;

    for (std::size_t i = 0; i < N; ++i)
    {
        if (i == j) continue;

        double term = 1.0 / (points[j] - points[i]);
        for (std::size_t k = 0; k < N; ++k)
        {
            if (k == i || k == j) continue;
            term *= (x - points[k]) / (points[j] - points[k]);
        }
        result += term;
    }

    return result;
}

/**
 * @brief Tensor-product Lagrange basis with Gauss–Legendre quadrature points.
 *
 * @tparam Order Polynomial order (number of nodes per dimension)
 * @tparam Dim Spatial dimension
 */
template <unsigned int Order, unsigned int Dim>
class Basis
{
private:
    vector<Order> _quadpoints;
    vector<Order> _quadweights;

public:
    /**
     * @brief Construct a tensor-product Lagrange basis on [start, end]^Dim.
     *
     * @param start Left interval bound (default 0)
     * @param end Right interval bound (default 1)
     */
    Basis(double start = 0.0, double end = 1.0)
    {
        GaussLegendre<Order> quad_1d(start, end);
        _quadpoints  = quad_1d.points();
        _quadweights = quad_1d.weights();
    }

    /// @return Reference to quadrature points
    const auto& quadpoints() const
    {
        return _quadpoints;
    }

    /// @return Reference to quadrature weights
    const auto& quadweights() const
    {
        return _quadweights;
    }

    /// @return Polynomial order
    constexpr unsigned int order() const
    {
        return Order;
    }

    /// @return Dimensionality of the basis
    constexpr unsigned int dimensions() const
    {
        return Dim;
    }

    /**
     * @brief Evaluate the tensor-product Lagrange basis expansion at a given
     * position.
     *
     * Works for both scalar (double) and vector-valued coefficient tensors.
     *
     * @tparam CoeffTensor Tensor type storing basis coefficients
     * @param coeffs Tensor of coefficients (rank must match Dim)
     * @param position Evaluation point in reference space
     * @return Interpolated field (double or vector)
     */
    template <typename CoeffTensor>
    [[nodiscard]]
    auto evaluate_basis(const CoeffTensor& coeffs, const vector<Dim>& position) const
    {
        static_assert(
            CoeffTensor::rank() == Dim,
            "Coefficient tensor rank must match spatial dimension"
        );

        using value_t       = typename CoeffTensor::value_type;
        using multi_index_t = typename CoeffTensor::multi_index_t;
        value_t sum{};

        multi_index_t multi_idx{};
        do
        {
            double prod = 1.0;
            for (unsigned int j = 0; j < Dim; ++j)
                prod *= lagrange_1d(
                    _quadpoints, static_cast<std::size_t>(multi_idx[j]), position[j]
                );
            sum = sum + (coeffs[multi_idx] * prod);
        } while (multi_idx.increment());

        return sum;
    }

    /**
     * @brief Project a function onto the reference tensor-product Lagrange basis.
     *
     * Works for both scalar and vector-valued functions.
     *
     * @tparam Func Function or functor type callable as `fun(position)`
     * @param fun Callable that accepts vector<double, Dim> and returns double
     * or vector
     * @return Coefficient tensor of the same type as function output
     */
    template <typename Func>
    [[nodiscard]]
    auto project_to_reference_basis(Func&& fun) const
    {
        // Determine the return type of the function
        using return_t = decltype(fun(std::declval<vector<Dim>>()));

        // Create a compile-time tensor of rank Dim, size Order along each axis
        using tensor_t = typename amr::containers::utils::types::tensor::
            hypercube_t<return_t, Order, Dim>;
        using multi_index_t = typename tensor_t::multi_index_t;

        tensor_t coeffs{};

        // Loop over all multi-indices
        multi_index_t multi_idx{};
        do
        {
            vector<Dim> position;
            for (unsigned int d = 0; d < Dim; ++d)
                position[d] = _quadpoints[multi_idx[d]];
            coeffs[multi_idx] = fun(position);
        } while (multi_idx.increment());

        return coeffs;
    }

    /**
     * @brief Create a 1D kernel vector φ by evaluating basis functions at a face
     * coordinate.
     *
     * For a given face coordinate along a specific dimension, this evaluates the 1D
     * basis functions at that fixed coordinate. This is used for projecting the
     * solution onto a face.
     *
     * @param face_coord The coordinate value along the dimension (e.g., 0.0 for left
     * face, 1.0 for right face)
     * @return constexpr vector of length Order containing φ evaluated at
     * face_coord
     */
    [[nodiscard]]
    constexpr vector<Order> create_face_kernel(double face_coord) const
    {
        vector<Order> phi;
        for (unsigned int i = 0; i < Order; ++i)
        {
            phi[i] = lagrange_1d(_quadpoints, i, face_coord);
        }

        return phi;
    }
};

} // namespace amr::Basis
#endif
