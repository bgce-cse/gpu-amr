#ifndef AMR_INCLUDED_STATIC_MATRIX
#define AMR_INCLUDED_STATIC_MATRIX

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <utility>
#include <type_traits>

#ifndef NDEBUG
#    define AMR_CONTAINERS_CHECKBOUNDS
#endif

#define LU_DETERTMINANT_TOL 1e-7

namespace amr::containers
{

template <typename T, std::integral auto N, std::integral auto M>
    requires std::is_same_v<decltype(N), decltype(M)>
class static_matrix
{
public:
    using value_type      = std::remove_cv_t<T>;
    using size_type       = std::common_type_t<decltype(M), decltype(N)>;
    using const_iterator  = value_type const*;
    using iterator        = value_type*;
    using const_reference = value_type const&;
    using reference       = value_type&;

    inline static constexpr auto      s_size_y = M;
    inline static constexpr auto      s_size_x = N;
    inline static constexpr size_type s_flat_size{ N * M };

    static_assert(std::is_trivially_copyable_v<T>);
    static_assert(std::is_standard_layout_v<T>);

    [[nodiscard]]
    constexpr static auto flat_size() noexcept -> size_type
    {
        return s_flat_size;
    }

    [[nodiscard]]
    constexpr static auto size_x() noexcept -> size_type
    {
        return s_size_x;
    }

    [[nodiscard]]
    constexpr static auto size_y() noexcept -> size_type
    {
        return s_size_y;
    }

    [[nodiscard]]
    constexpr auto operator[](const size_type j, const size_type i) noexcept -> reference
    {
        return const_cast<reference>(std::as_const(*this).operator[](i,j));
    }

    [[nodiscard]]
    constexpr auto operator[](const size_type j, const size_type i) const noexcept
        -> const_reference
    {
#ifdef AMR_CONTAINERS_CHECKBOUNDS
        assert_in_bounds(j, i);
#endif
        return data_[j * N + i];
    }

    [[nodiscard]]
    constexpr auto cbegin() const noexcept -> const_iterator
    {
        return std::cbegin(data_);
    }

    [[nodiscard]]
    constexpr auto cend() const noexcept -> const_iterator
    {
        return std::cend(data_);
    }

    [[nodiscard]]
    constexpr auto begin() const noexcept -> const_iterator
    {
        return std::begin(data_);
    }

    [[nodiscard]]
    constexpr auto end() const noexcept -> const_iterator
    {
        return std::end(data_);
    }

    [[nodiscard]]
    constexpr auto begin() noexcept -> iterator
    {
        return std::begin(data_);
    }

    [[nodiscard]]
    constexpr auto end() noexcept -> iterator
    {
        return std::end(data_);
    }

#ifdef AMR_CONTAINERS_CHECKBOUNDS
    constexpr auto assert_in_bounds(size_type const j, size_type const i) const noexcept
        -> void
    {
        assert(i < s_size_x);
        assert(j < s_size_y);
        if constexpr (std::is_signed_v<size_type>)
        {
            assert(i >= size_type{});
            assert(j >= size_type{});
        }
    }
#endif

    constexpr auto operator<=>(static_matrix const&) const noexcept = default;

private:
    // TODO: Alignment, maybe padding
    value_type data_[s_flat_size];
};

template <typename T, std::integral auto N, std::integral auto M>
using row_matrix = static_matrix<T, 1, N>;

template <typename T, std::integral auto N, std::integral auto M>
using column_matrix = static_matrix<T, M, 1>;

template <typename T, std::integral auto N, std::integral auto M>
std::ostream& operator<<(std::ostream& os, static_matrix<T, N, M> const& mat)
{
    using matrix_t   = static_matrix<T, N, M>;
    using size_type  = typename matrix_t::size_type;
    using value_type = typename matrix_t::value_type;
    if constexpr (std::is_floating_point_v<value_type>)
    {
        os << std::fixed;
        os << std::setprecision(4);
    }
    os << "[";
    for (size_type j = 0; j != matrix_t::size_y(); ++j)
    {
        os << "\n [ ";
        for (size_type i = 0; i != matrix_t::size_x(); ++i)
        {
            os << mat[j, i] << ' ';
        }
        os << "]";
    }
    os << "\n]\n";
    if constexpr (std::is_floating_point_v<value_type>)
    {
        os << std::defaultfloat;
    }
    return os;
}

template <typename T, std::integral auto M, std::integral auto K, std::integral auto N>
    requires std::is_same_v<decltype(M), decltype(N)> &&
             std::is_same_v<decltype(M), decltype(K)> &&
             std::is_default_constructible_v<static_matrix<T, M, N>>
[[nodiscard]]
constexpr auto matrix_mul(
    static_matrix<T, M, K> const& mat1,
    static_matrix<T, K, N> const& mat2
) noexcept -> static_matrix<T, M, N>
{
    static_matrix<T, M, N> ret{};

    for (decltype(M) j = 0; j != M; ++j)
    {
        for (decltype(K) k = 0; k != K; ++k)
        {
            for (decltype(N) i = 0; i != N; ++i)
            {
                ret[j, i] += mat1[j, k] * mat2[k, i];
            }
        }
    }
    return ret;
}

} // namespace amr::containers

#endif // AMR_INCLUDED_STATIC_MATRIX
