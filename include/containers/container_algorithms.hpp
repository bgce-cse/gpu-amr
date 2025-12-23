#ifndef AMR_INCLUDED_CONTAINER_FACTORIES
#define AMR_INCLUDED_CONTAINER_FACTORIES

#include "container_manipulations.hpp"
#include "container_utils.hpp"
#include "static_shape.hpp"
#include "static_vector.hpp"
#include <concepts>

namespace amr::containers::algorithms
{

namespace tensor
{

// TODO: Update to shaped for rather than multiindex iteration
template <std::integral auto Rank, typename T, std::integral auto Size>
[[nodiscard]]
constexpr auto cartesian_expansion(static_vector<T, Size> const& v) noexcept
    -> utils::types::tensor::hypercube_t<static_vector<T, Rank>, Size, Rank>
{
    using index_t     = typename std::remove_cvref_t<decltype(v)>::index_t;
    using hypercube_t = utils::types::tensor::
        hypercube_t<static_vector<T, index_t{ Rank }>, Size, index_t{ Rank }>;
    using multi_index_t = typename hypercube_t::multi_index_t;
    static_assert(std::is_same_v<typename multi_index_t::index_t, index_t>);
    auto ret = hypercube_t{};
    auto idx = multi_index_t{};
    do
    {
        for (index_t d{}; d != index_t{ Rank }; ++d)
        {
            ret[idx][d] = v[idx[d]];
        }
    } while (idx.increment());

    return ret;
}

template <concepts::StaticContainer A, concepts::StaticContainer B>
[[nodiscard]]
constexpr auto tensor_product(A const& a, B const& b) noexcept -> utils::types::tensor::
    tensor_product_result_t<std::remove_cvref_t<A>, std::remove_cvref_t<B>>
{
    using a_t   = std::remove_cvref_t<A>;
    using b_t   = std::remove_cvref_t<B>;
    using ret_t = utils::types::tensor::tensor_product_result_t<a_t, b_t>;
    static_assert(
        std::tuple_size_v<std::remove_cvref_t<decltype(ret_t::sizes())>> ==
        a_t::rank() + b_t::rank()
    );

    ret_t ret{};
    /*
    using lc_t = control::loop_control<ret_t, 0, ret_t::sizes(), 1>;
    manipulators::shaped_for<lc_t>(
        [&ret](auto const& a, auto const& b, auto const& idxs)
        {
            ret[idxs] = a[std::span<typename ret_t::index_t const, a_t::rank()>{
                            idxs.data(), a_t::rank() }] *
                        b[std::span<typename ret_t::index_t const, b_t::rank()>{
                            idxs.data() + a_t::rank(), b_t::rank() }];
        },
        t1,
        t2
    );
    */
    using index_t              = typename ret_t::index_t;
    constexpr auto outter_size = static_cast<index_t>(A::shape_t::elements());
    constexpr auto inner_size  = static_cast<index_t>(B::shape_t::elements());
    for (auto i = index_t{}; i != outter_size; ++i)
    {
        for (auto j = index_t{}; j != inner_size; ++j)
        {
            ret[i * inner_size + j] = a[i] * b[j];
        }
    }
    return ret;
}

template <
    std::floating_point F,
    std::integral auto  Rank,
    std::integral auto  Order,
    std::integral auto  Dofs>
[[nodiscard]]
constexpr auto evaluate_basis(
    utils::types::tensor::hypercube_t<static_vector<F, Dofs>, Order, Rank> const& coeffs,
    static_vector<F, Rank> const&                                                 x,
    static_vector<F, Order> const& quad_points
) noexcept -> F
{
    // TODO
}

/// Apply a vector to a tensor element-wise along a specified dimension (einsum-like
/// operation)
///
/// Multiplies a tensor by a vector along a specific dimension. The vector must have
/// size equal to tensor.size(DimVec). This broadcasts the vector along that dimension.
///
/// Example:
///   For a tensor of shape [3, 4, 5] and vector of size 3, applying along dim 0
///   multiplies tensor[i,j,k] by vec[i] for all j,k.
///
/// Usage:
///   auto vec = static_vector<double, 3>{1.0, 2.0, 3.0};
///   auto tensor = static_tensor<double, Layout<3,4,5>>{...};
///   auto result = einsum_apply<0>(tensor, vec);
///
/// @tparam DimVec The dimension along which to apply the vector
/// @param tensor The input tensor
/// @param vec The vector to apply (must match size of dimension DimVec)
/// @return A new tensor with vector applied element-wise along the dimension
template <
    std::integral auto        DimVec,
    concepts::StaticContainer Tensor,
    concepts::Vector          Vec>
    requires(DimVec < std::remove_cvref_t<Tensor>::rank())
[[nodiscard]]
constexpr auto einsum_apply(Tensor const& tensor, Vec const& vec) noexcept
    -> std::remove_cvref_t<Tensor>
{
    using tensor_t = std::remove_cvref_t<Tensor>;
    using vec_t    = std::remove_cvref_t<Vec>;
    using value_t  = typename tensor_t::value_type;
    using index_t  = typename tensor_t::index_t;
    // Verify vector size matches tensor dimension
    static_assert(
        vec_t::elements() == tensor_t::size(index_t{ DimVec }),
        "Vector size must match tensor dimension size"
    );
    // Check value types compatibility
    if constexpr (concepts::Vector<value_t>)
    {
        // Tensor has vector values (like static_vector<double, 3>)
        // Vec must contain the scalar type
        using scalar_t = typename value_t::value_type;
        static_assert(
            std::is_same_v<typename vec_t::value_type, scalar_t>,
            "For vector-valued tensors, Vec must contain scalars matching the tensor's "
            "scalar type"
        );
    }
    else
    {
        // Tensor has scalar values
        // Vec must contain the same scalar type
        static_assert(
            std::is_same_v<typename vec_t::value_type, value_t>,
            "For scalar-valued tensors, Vec must match the tensor's value type"
        );
    }
    auto result = tensor_t{};
    // Iterate over all elements and multiply by corresponding vector element
    auto multi_idx = typename tensor_t::multi_index_t{};
    do
    {
        // This works for both cases:
        // - If value_t is scalar: scalar * scalar
        // - If value_t is vector: vector * scalar (element-wise multiplication)
        result[multi_idx] = tensor[multi_idx] * vec[multi_idx[index_t{ DimVec }]];
    } while (multi_idx.increment());
    return result;
}

/// Apply a function that combines tensor and vector values along a dimension
///
/// Applies a binary operation between tensor elements and vector values,
/// indexed by a specified dimension. Flexible for various einsum-like operations.
///
/// @tparam DimVec The dimension along which to apply the operation
/// @param tensor The input tensor
/// @param vec The vector to apply
/// @param op Binary operation (tensor_value, vec_value) -> result_value
/// @return Result tensor with operation applied
// Specialized version for vector-valued tensors with scalar kernel vector
// Single general einsum_apply that handles both scalar and vector-valued tensors
template <
    std::integral auto        DimVec,
    concepts::StaticContainer Tensor,
    concepts::Vector          Vec>
    requires(DimVec < std::remove_cvref_t<Tensor>::rank())
[[nodiscard]]
constexpr auto einsum_apply_vector(Tensor const& tensor, Vec const& vec) noexcept
    -> std::remove_cvref_t<Tensor>
{
    using tensor_t = std::remove_cvref_t<Tensor>;
    using vec_t    = std::remove_cvref_t<Vec>;
    using value_t  = typename tensor_t::value_type;
    using index_t  = typename tensor_t::index_t;
    static_assert(
        std::is_same_v<typename vec_t::value_type, value_t>,
        "Vector and tensor must have the same value type"
    );
    // Verify vector size matches tensor dimension
    static_assert(
        vec_t::elements() == tensor_t::size(index_t{ DimVec }),
        "Vector size must match tensor dimension size"
    );
    auto result = tensor_t{};
    // Iterate over all elements and multiply by corresponding vector element
    {
        auto multi_idx = typename tensor_t::multi_index_t{};
        do
        {
            result[multi_idx] = tensor[multi_idx] * vec[multi_idx[index_t{ DimVec }]];
        } while (multi_idx.increment());
    }
    return result;
}

template <
    std::integral auto        DimVec,
    concepts::StaticContainer Tensor,
    concepts::Vector          Vec>
    requires(DimVec < std::remove_cvref_t<Tensor>::rank())
[[nodiscard]]
constexpr auto contract(Tensor const& tensor, Vec const& vec) noexcept
{
    using tensor_t = std::remove_cvref_t<Tensor>;
    using value_t  = typename tensor_t::value_type;
    using index_t  = typename tensor_t::index_t;
    using vec_t    = std::remove_cvref_t<Vec>;
    static_assert(Vec::elements() == tensor_t::size(index_t{ DimVec }));
    // Verify vector and tensor types are compatible
    // They can either have the same value type, or tensor has vector values
    // and vec has the scalar type
    if constexpr (concepts::Vector<value_t>)
    {
        using scalar_t = typename value_t::value_type;
        static_assert(
            std::is_same_v<typename vec_t::value_type, scalar_t>,
            "For vector-valued tensors, Vec must contain scalars matching the tensor's "
            "scalar type"
        );
    }
    else
    {
        static_assert(
            std::is_same_v<typename vec_t::value_type, value_t>,
            "For scalar-valued tensors, Vec must match the tensor's value type"
        );
    }
    // Verify this is a hypercube tensor (all dimensions equal)
    [&]<auto... Is>(std::index_sequence<Is...>)
    {
        static_assert(
            (... && (tensor_t::size(index_t{ Is }) == tensor_t::size(index_t{ 0 }))),
            "contract requires a hypercube tensor (all dimensions must be equal)"
        );
    }(std::make_index_sequence<tensor_t::rank()>{});
    // Step 1: Element-wise multiply
    auto multiplied = einsum_apply<DimVec>(tensor, vec);
    // Step 2: Create result tensor with reduced rank using hypercube
    // For a hypercube of rank N with size S, the result is a hypercube of rank N-1 with
    // size S
    constexpr auto rank = tensor_t::rank();
    constexpr auto size = tensor_t::size(index_t{ 0 }); // All dims same for hypercube
    using result_hypercube_t = utils::types::tensor::hypercube_t<value_t, size, rank - 1>;
    auto result              = result_hypercube_t::zero();
    // Iterate through result indices and sum over the contracted dimension

    {
        auto result_idx = typename result_hypercube_t::multi_index_t{};
        do
        {
            // For each result index, we need to map it to the full tensor space
            // and sum along the contracted dimension
            bool first = true;
            for (auto sum_val = index_t{}; sum_val < tensor_t::size(index_t{ DimVec });
                 ++sum_val)
            {
                // Construct full index by inserting sum_val at position DimVec
                auto full_idx = typename tensor_t::multi_index_t{};
                // Copy indices before DimVec
                for (auto d = index_t{}; d < index_t{ DimVec }; ++d)
                {
                    full_idx[d] = result_idx[d];
                }
                // Set the contracted dimension
                full_idx[index_t{ DimVec }] = sum_val;
                // Copy indices after DimVec (shifted by 1 in result)
                for (auto d = index_t{ DimVec }; d < index_t{ rank - 1 }; ++d)
                {
                    full_idx[d + 1] = result_idx[d];
                }
                if (first)
                {
                    result[result_idx] = multiplied[full_idx];
                    first              = false;
                }
                else
                {
                    result[result_idx] = result[result_idx] + multiplied[full_idx];
                }
            }
        } while (result_idx.increment());
    }
    return result;
}

template <
    concepts::ContractionIndexSet auto CIS,
    concepts::StaticContainer          A,
    concepts::StaticContainer          B>
[[nodiscard, deprecated("Not implemented")]]
constexpr auto
    contraction([[maybe_unused]] A const& a, [[maybe_unused]] B const& b) noexcept
    -> utils::types::tensor::
        tensor_contraction_result_t<std::remove_cvref_t<A>, std::remove_cvref_t<B>, CIS>
{
    using a_t                                   = std::remove_cvref_t<A>;
    using b_t                                   = std::remove_cvref_t<B>;
    static constexpr auto contraction_index_set = CIS;
    using ret_t                                 = utils::types::tensor::
        tensor_contraction_result_t<a_t, b_t, contraction_index_set>;
    // using reduction_loop_shape_t = containers::static_shape<>;

    ret_t ret{};
    containers::manipulators::apply(
        ret,
        [](auto& out, auto const& idxs)
        {
            out[idxs] = 1;
            utility::error_handling::assert_unreachable();
        }
    );
    return ret;
}

template <std::size_t N, typename TensorType>
constexpr auto tensor_power(TensorType const& tensor)
{
    static_assert(N >= 1, "Tensor power must be at least 1");
    if constexpr (N == 1)
        return tensor;
    else
        return tensor_product(tensor, tensor_power<N - 1>(tensor));
}

template <typename Derivative, typename Flux>
auto derivative_contraction(
    Derivative const&  derivative,
    Flux const&        flux,
    std::integral auto dim
)
{
    using multi_index_t = typename Flux::multi_index_t;
    using value_type    = typename Flux::value_type; // This is static_vector<double, 4>
    using index_t       = typename Flux::index_t;

    static constexpr auto Order = std::remove_cvref_t<Derivative>::size(0);

    auto const& flux_component = flux;
    Flux        result         = Flux::zero();

    // Iterate over all result indices
    auto idx = multi_index_t{};
    do
    {
        value_type sum{};

        auto const i_dim = idx[dim];

        for (index_t a = 0; a < Order; ++a)
        {
            auto flux_idx = idx;
            flux_idx[dim] = a;

            sum = sum + derivative[i_dim, a] * flux_component[flux_idx];
        }

        result[idx] = sum;
    } while (idx.increment());

    return result;
}

} // namespace tensor

} // namespace amr::containers::algorithms

#endif // AMR_INCLUDED_CONTAINER_FACTORIES
