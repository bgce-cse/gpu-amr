#ifndef AMR_INCLUDED_CONTAINER_MANIPULATIONS
#define AMR_INCLUDED_CONTAINER_MANIPULATIONS

#include "container_concepts.hpp"
#include "container_utils.hpp"
#include "loop_control.hpp"
#include "static_tensor.hpp"
#include "static_vector.hpp"
#include <algorithm>
#include <array>
#include <concepts>
#include <functional>
#include <type_traits>

namespace amr::containers::manipulators
{

template <concepts::Container C>
constexpr auto fill(C& c, typename C::value_type const& v) noexcept -> void
{
    std::ranges::fill(c, v);
}

template <concepts::Container C, typename Fn, typename... Args>
    requires std::invocable<Fn, Args...> &&
             std::
                 convertible_to<std::invoke_result_t<Fn, Args...>, typename C::value_type>
constexpr auto fill(C& c, Fn&& fn, Args&&... args) noexcept(
    std::is_nothrow_invocable_r_v<typename C::value_type, Fn, Args...>
) -> void
{
    for (auto it = std::begin(c); it != std::end(c); ++it)
    {
        *it = std::invoke(fn, std::forward<Args>(args)...);
    }
}

namespace detail
{

// TODO: Compare with and without [[gnu::always_inline, gnu::flatten]]
template <
    concepts::LoopControl Loop_Control,
    std::integral auto    I,
    std::integral... Indices>
[[gnu::always_inline, gnu::flatten]]
constexpr auto shaped_for_impl(auto&& fn, Indices... idxs, auto&&... args) noexcept
    -> void
{
    using loop_t = Loop_Control;
    using rank_t = typename loop_t::rank_t;

    if constexpr (I == loop_t::rank())
    {
        static_assert(std::invocable<decltype(fn), decltype(args)..., decltype(idxs)...>);
        std::invoke(
            std::forward<decltype(fn)>(fn), std::forward<decltype(args)>(args)..., idxs...
        );
    }
    else
    {
        for (auto i = loop_t::start(I); i != loop_t::end(I); i += loop_t::stride(I))
        {
            shaped_for_impl<loop_t, I + rank_t{ 1 }, Indices..., decltype(i)>(
                std::forward<decltype(fn)>(fn),
                idxs...,
                i,
                std::forward<decltype(args)>(args)...
            );
        }
    }
}

template <
    concepts::LoopControl Loop_Control,
    std::integral auto    I,
    std::integral         Index_Type>
[[gnu::always_inline, gnu::flatten]]
constexpr auto shaped_for_impl(
    auto&&                                        fn,
    std::array<Index_Type, Loop_Control::rank()>& idxs,
    auto&&... args
) noexcept -> void
{
    using loop_t = Loop_Control;
    using rank_t = typename loop_t::rank_t;

    if constexpr (I == loop_t::rank())
    {
        static_assert(std::invocable<decltype(fn), decltype(args)..., decltype(idxs)>);
        std::invoke(
            std::forward<decltype(fn)>(fn), std::forward<decltype(args)>(args)..., idxs
        );
    }
    else
    {
        for (idxs[I] = loop_t::start(I); idxs[I] != loop_t::end(I);
             idxs[I] += loop_t::stride(I))
        {
            shaped_for_impl<loop_t, I + rank_t{ 1 }, Index_Type>(
                std::forward<decltype(fn)>(fn),
                idxs,
                std::forward<decltype(args)>(args)...
            );
        }
    }
}

} // namespace detail

template <concepts::LoopControl Loop_Control>
[[gnu::always_inline, gnu::flatten]]
constexpr auto shaped_for(auto&& fn, auto&&... args) noexcept -> void
{
    using loop_t = Loop_Control;
    using rank_t = typename loop_t::rank_t;
    std::array<typename loop_t::index_t, loop_t::rank()> idxs{};
    detail::shaped_for_impl<Loop_Control, rank_t{}>(
        std::forward<decltype(fn)>(fn), idxs, std::forward<decltype(args)>(args)...
    );
}

template <concepts::LoopControl Loop_Control>
[[gnu::always_inline, gnu::flatten]]
constexpr auto for_each(auto&& a, auto&& fn, auto&&... args) noexcept -> void
    requires concepts::StaticContainer<std::remove_cvref_t<decltype(a)>>
{
    static_assert(std::is_same_v<
                  typename std::remove_cvref_t<decltype(a)>::shape_t,
                  typename Loop_Control::shape_t>);
    shaped_for<Loop_Control>(
        std::forward<decltype(fn)>(fn),
        std::forward<decltype(a)>(a),
        std::forward<decltype(args)>(args)...
    );
}

[[gnu::always_inline, gnu::flatten]]
constexpr auto apply(auto&& a, auto&& fn, auto&&... args) noexcept -> void
    requires concepts::StaticContainer<std::remove_cvref_t<decltype(a)>>
{
    using s_t  = typename std::remove_cvref_t<decltype(a)>::shape_t;
    using lc_t = control::loop_control<s_t, 0, s_t::sizes(), 1>;
    for_each<lc_t>(
        std::forward<decltype(a)>(a),
        std::forward<decltype(fn)>(fn),
        std::forward<decltype(args)>(args)...
    );
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

template <typename Tensor, typename Vec>
constexpr auto prepend_shape(Tensor const&, Vec const& vec)
{
    constexpr auto                    rank      = Tensor::rank();
    constexpr auto                    size      = Vec::elements();
    auto                              old_sizes = Tensor::sizes();
    std::array<std::size_t, rank + 1> new_sizes{};
    new_sizes[0] = size;
    for (std::size_t i = 0; i < rank; ++i)
        new_sizes[i + 1] = old_sizes[i];
    return new_sizes;
}

} // namespace amr::containers::manipulators

#endif // AMR_INCLUDED_CONTAINER_MANIPULATIONS
