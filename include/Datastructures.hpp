#pragma once

#include <cassert>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#define DEBUG_MATRIX_ACCESS (false)
#if DEBUG_MATRIX_ACCESS
#include <exception>
#include <sstream>
#include <stdexcept>
#endif

/**
 * @brief General 2D data structure around std::vector, in column
 * major format.
 *
 */
template <typename T>
class Matrix
{

  public:
    Matrix() = default;

    /**
     * @brief Constructor with initial value
     *
     * @param[in] number of elements in x direction
     * @param[in] number of elements in y direction
     * @param[in] initial value for the elements
     *
     */
    Matrix(int num_cols, int num_rows, T init_val)
        : Matrix(num_cols, num_rows)
    {
        std::fill(_container.begin(), _container.end(), init_val);
    }

    /**
     * @brief Constructor without an initial value.
     *
     * @param[in] number of elements in x direction
     * @param[in] number of elements in y direction
     *
     */
    Matrix(int num_cols, int num_rows)
        : _num_cols(num_cols)
        , _num_rows(num_rows)
    {
        assert(_num_rows > 0);
        assert(_num_cols > 0);
        const auto flat_size = num_cols * num_rows;
        assert(flat_size >= 0);
        _container.resize(static_cast<std::size_t>(flat_size));
    }

    /**
     * @brief Element access and modify using index
     *
     * @param[in] x index
     * @param[in] y index
     * @param[out] reference to the value
     */
    [[nodiscard]]
    inline T& operator()(int i, int j)
    {
        // Scott Meyers Effective C++
        //  Item 3: Use const whenever possible,
        //  section: Avoiding Duplication in const and Non-const Member
        //  Functions
        return const_cast<T&>(std::as_const(*this).operator()(i, j));
    }

    /**
     * @brief Element access using index
     *
     * @param[in] x index
     * @param[in] y index
     * @param[out] value of the element
     */
    [[nodiscard]]
    inline T const& operator()(int i, int j) const
    {
#if DEBUG_MATRIX_ACCESS
        if (i < 0 || i >= _num_cols || j < 0 || j >= _num_rows) [[unlikely]]
        {
            std::stringstream ss;
            ss << "Out of bounds memory access for " << this << '\n'
               << "\tAccess i = " << i << ", range: [0, " << _num_cols << ")\n"
               << "\tAccess j = " << j << ", range: [0, " << _num_rows << ")\n";
            throw std::out_of_range(ss.str());
        }
#endif
        assert(i >= 0 && i < _num_cols);
        assert(j >= 0 && j < _num_rows);
        const auto flat_idx = static_cast<std::size_t>(_num_cols * j + i);
        assert(flat_idx < _container.size());
        return _container[flat_idx];
    }

    /**
     * @brief Pointer representation of underlying data
     *
     * @param[out] pointer to the beginning of the vector
     */
    [[nodiscard]]
    inline const T* data() const noexcept
    {
        return _container.data();
    }
    [[nodiscard]]
    inline T* data() noexcept
    {
        return _container.data();
    }

    /**
     * @brief Access of the size of the structure
     *
     * @param[out] size of the data structure
     */
    [[nodiscard]]
    inline int size() const noexcept
    {
        return _container.size();
    }

    /// get the given row of the matrix
    [[nodiscard]]
    std::vector<double> get_row(int row) const
    {
        std::vector<T> row_data(_num_cols, -1);
        for (int i = 0; i < _num_cols; ++i)
        {
            row_data.at(i) = _container.at(i + _num_cols * row);
        }
        return row_data;
    }

    /// get the given column of the matrix
    [[nodiscard]]
    std::vector<double> get_col(int col) const
    {
        std::vector<T> col_data(_num_rows, -1);
        for (int i = 0; i < _num_rows; ++i)
        {
            col_data.at(i) = _container.at(col + i * _num_cols);
        }
        return col_data;
    }

    /// set the given column of matrix to given vector
    void set_col(const std::vector<double>& vec, int col)
    {
        for (int i = 0; i < _num_rows; ++i)
        {
            _container.at(col + i * _num_cols) = vec.at(i);
        }
    }

    /// set the given row of matrix to given vector
    void set_row(const std::vector<double>& vec, int row)
    {
        for (int i = 0; i < _num_cols; ++i)
        {
            _container.at(i + row * _num_cols) = vec.at(i);
        }
    }

    /// get the number of elements in x direction
    [[nodiscard]]
    inline int num_cols() const noexcept
    {
        return _num_cols;
    }

    /// get the number of elements in y direction
    [[nodiscard]]
    inline int num_rows() const noexcept
    {
        return _num_rows;
    }

    [[nodiscard]] auto cbegin() const noexcept -> auto
    {
        return std::cbegin(_container);
    }

    [[nodiscard]] auto cend() const noexcept -> auto
    {
        return std::cend(_container);
    }

    [[nodiscard]] auto begin() const noexcept -> auto
    {
        return std::cbegin(_container);
    }

    [[nodiscard]] auto end() const noexcept -> auto
    {
        return std::cend(_container);
    }

    [[nodiscard]] auto begin() noexcept -> auto
    {
        return std::begin(_container);
    }

    [[nodiscard]] auto end() noexcept -> auto
    {
        return std::end(_container);
    }

  private:
    /// Number of elements in x direction
    int _num_cols;
    /// Number of elements in y direction
    int _num_rows;

    /// Data container
    std::vector<T> _container;
};
