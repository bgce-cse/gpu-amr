#ifndef AMR_INLUDED_STATIC_MATRIX
#define AMR_INLUDED_STATIC_MATRIX

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
#    define AMR_CONTANERS_CHECKBOUNDS
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
#ifdef AMR_CONTANERS_CHECKBOUNDS
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

#ifdef AMR_CONTANERS_CHECKBOUNDS
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
    if (std::is_floating_point_v<value_type>)
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
    if (std::is_floating_point_v<value_type>)
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

/*
efficient LU decompoition of N by N matrix M
returns:
  bool: det(M) != 0 //If matrix is inversible or not
  F: det(M)
  static_matrix LU: L is lower triangular with unit diagonal (implicit)
             U is upped diagonal, including diagonal elements

*/
template <std::floating_point F, std::size_t N>
    requires(N > 1)
[[nodiscard]]
constexpr auto PII_LUDecomposition(static_matrix<F, N, N>& src)
    -> std::tuple<bool, F, std::array<decltype(N), N>>
{
    /*
    source:
    http://web.archive.org/web/20150701223512/http://download.intel.com/design/PentiumIII/sml/24504601.pdf

    Factors "_Source" matrix into out=LU where L is lower triangular and U is
    upper triangular. The matrix is overwritten by LU with the diagonal elements
    of L (which are unity) not stored. This must be a square n x n matrix.
    */

    F det = F{ 1 };

    // Initialize the pointer vector.
    std::array<decltype(N), N> r_idx{}; // row index
    std::iota(std::begin(r_idx), std::end(r_idx));

    // LU factorization.
    for (decltype(N) p = 0; p != N - 1; ++p)
    {
        // Find pivot element.
        for (decltype(N) j = p + 1; j != N; ++j)
        {
            if (std::abs(src[r_idx[j], p]) > std::abs(src[r_idx[p], p]))
            {
                // Swap the index for the p pivot row if necessary.;
                std::swap(r_idx[j], r_idx[p]);
                det = -det;
                // RIdx[p] now has the index of the row to consider the pth
            }
        }

        if (out(r_idx[p], p) == 0)
        {
            // The matrix is singular. //or not invertible by this method until
            // fixed (no permutations)
            return { false, 0.0, {} };
        }
        // Multiply the diagonal elements.
        det *= src[r_idx[p], p];

        // Form multiplier.
        for (decltype(N) j = p + 1; j != N; ++j)
        {
            src[r_idx[j], p] /= src[r_idx[p], p];
            // Eliminate [p].
            for (decltype(N) i = p + 1; i != N; ++i)
            {
                src[r_idx[j], i] -= src[r_idx[j], p] * src[r_idx[p], i];
            }
        }
    }
    det *= src[r_idx[N - 1], N - 1]; // multiply last diagonal element

    const auto ri = r_idx;

    // reorder output for simplicity
    for (decltype(N) j = 0; j != N; ++j)
    {
        if (j != r_idx[j])
        {
            for (decltype(N) i = 0; i != N; ++i)
            {
                std::swap(src[j, i], src[r_idx[j], i]);
            }
            std::swap(
                r_idx[j],
                r_idx[decltype(N){ std::find(std::begin(r_idx), std::end(r_idx), j) -
                                   std::begin(r_idx) }]
            );
        }
    }
    // F.21
    // https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#f21-to-return-multiple-out-values-prefer-returning-a-struct-or-tuple
    bool invertible = std::abs(det >= LU_DETERTMINANT_TOL);
    return { invertible, invertible ? det : NAN, std::move(ri) };
}

/**
 * \brief Mutates matrix to its reduced row echelon form
 * Can be used to solve systems of linear equations with an augmented matrix
 * or to invert a matrix M : ( M | I ) -> ( I | M^-1 )
 * \param src Input matrix
 * \return Bool indicating whether or not the matrix is invertible and the
 * transformation succeeded
 *
 * \note Uses Gauss-Jordan elimination with partial pivoting
 *       Mutates input
 */
template <std::floating_point T, std::size_t M, std::size_t N>
    requires(M <= N)
[[nodiscard]]
constexpr auto RREF(static_matrix<T, M, N>& src) -> bool
{
    std::array<size_t, N> r_idx{}; // row index
    std::iota(std::begin(r_idx), std::end(r_idx), 0);

    for (decltype(M) p = 0; p != M; ++p)
    {
        for (decltype(M) j = p + 1; j < M; ++j)
        {
            if (std::abs(src[r_idx[p], p]) < std::abs(src[r_idx[j], p]))
            {
                std::swap(r_idx[p], r_idx[j]);
            }
        }

        if (src[r_idx[p], p] <= LU_DETERTMINANT_TOL) return false; // matrix is singular

        for (decltype(N) i = p + 1; i < N; ++i)
        {
            src[r_idx[p], i] /= src[r_idx[p], p];
        }
        src[r_idx[p], p] = 1;

        for (decltype(M) j = 0; j != M; ++j)
        {
            if (j != p)
            {
                for (decltype(N) i = p + 1; i < N; ++i)
                { // p+1 to avoid removing each rows' scale factor
                    src(r_idx[j], i) -= src(r_idx[p], i) * src(r_idx[j], p);
                }
                src(r_idx[j], p) = 0;
            }
        }
    }

    // reorder matrix
    for (decltype(M) j = 0; j != M; ++j)
    {
        if (j != r_idx[j])
        {
            for (decltype(N) i = 0; i != N; ++i)
            {
                std::swap(src[j, i], src[r_idx[j], i]);
            }
            std::swap(
                r_idx[j],
                r_idx[decltype(N){ std::find(std::begin(r_idx), std::end(r_idx), j) -
                                   std::begin(r_idx) }]
            );
        }
    }
    return true;
}

} // namespace amr::containers

#endif // AMR_INCLUDED_STATIC_MATRIX
