#pragma once

#include <cstddef>
#include <stdexcept>
#include <type_traits>

namespace lm {

template <typename M, typename Enable = void>
struct matrix_traits {

    typedef typename M::value_type value_type;
    typedef M matrix_type;

    static const value_type& cell(const matrix_type& m, size_t row, size_t col) {
        return m.cell(row, col);
    }

    static value_type& cell(matrix_type& m, size_t row, size_t col) {
        return m.cell(row, col);
    }

    static size_t rows(const matrix_type& m) {
        return m.rows();
    }

    static size_t cols(const matrix_type& m) {
        return m.cols();
    }

    static void swap_row(matrix_type& m, size_t n1, size_t n2) {
        m.swap_row(n1, n2);
    }

};

template <typename T>
struct matrix_traits<T, typename std::enable_if< std::is_array<typename std::remove_reference<T>::type>::value >::type> {

    typedef typename std::remove_reference<T>::type array_type;

    enum {
        Rows = std::extent<array_type, 0>::value,
        Cols = std::extent<array_type, 1>::value
    };

    typedef typename std::remove_all_extents< typename std::remove_reference<T>::type >::type value_type;
    typedef typename std::conditional< std::is_lvalue_reference<T>::value,
        T,
        typename std::add_lvalue_reference<T>::type
    >::type matrix_type;

    static value_type& cell(matrix_type m, size_t row, size_t col) {
        return m[row][col];
    }

    static size_t rows(const matrix_type m) {
        return Rows;
    }

    static size_t cols(const matrix_type m) {
        return Cols;
    }

    static void swap_row(matrix_type m, size_t n1, size_t n2) {
        for (size_t i = 0; i < cols(m); i++) {
            std::swap(cell(m, n1, i), cell(m, n2, i));
        }
    }

};

template <typename V>
struct vector_traits {

    typedef typename V::value_type value_type;
    typedef V vector_type;

    static void resize(vector_type& vec, size_t new_size) {
        vec.resize(new_size);
    }

    static size_t size(const vector_type& vec) {
        return vec.size();
    }

};

template <typename T, size_t Size>
struct vector_traits<T[Size]> {

    typedef T value_type;
    typedef T vector_type[Size];

    static void resize(vector_type& vec, size_t new_size) {
        if (new_size > Size) {
            throw std::length_error("array is too small");
        }
    }

    static size_t size(const vector_type& vec) {
        return Size;
    }

};


template <typename Matrix, typename Vec,
          typename MatrixTraits = matrix_traits<Matrix>,
          typename VectorTraits = vector_traits<Vec>>
class permutation_matrix {
public:

    typedef typename MatrixTraits::value_type value_type;

    permutation_matrix(Matrix m) : _m(m), _c(0) {
        init_permutation();
    }

    permutation_matrix(Matrix m, Vec p) : _m(m), _p(p), _c(0) {
        init_permutation();
    }

    size_t rows() const {
        return MatrixTraits::rows(_m);
    }

    size_t cols() const {
        return MatrixTraits::cols(_m);
    }

    const value_type& cell(size_t row, size_t col) const {
        return MatrixTraits::cell(_m, _p[row], col);
    }

    value_type& cell(size_t row, size_t col) {
        return MatrixTraits::cell(_m, _p[row], col);
    }

    void swap_row(size_t n1, size_t n2) {
        if (n1 == n2) {
            return;
        }

        std::swap(_p[n1], _p[n2]);
        ++_c;
    }

    size_t permutation_count() const {
        return _c;
    }


private:

    void init_permutation() {
        size_t l = rows();

        VectorTraits::resize(_p, l);
        for (size_t i = 0; i < l; i++) {
            _p[i] = i;
        }
    }

    Matrix _m;
    Vec _p;
    size_t _c;


};


template <typename Matrix, typename MatrixTraits = matrix_traits<Matrix>>
std::pair<size_t, bool> find_lu_pivot(Matrix& m, const size_t n) {

    size_t l = MatrixTraits::rows(m);

    std::pair<size_t, bool> result(0, false);
    for (size_t i = n; i < l; i++) {

        const typename MatrixTraits::value_type& v = MatrixTraits::cell(m, i, n);
        if (v == 0) {
            continue;
        }

        if (!result.second) {
            result.first = i;
            result.second = true;
            continue;
        }

        if (std::abs(v) < std::abs(MatrixTraits::cell(m, result.first, n))) {
            result.first = i;
        }

    }
    return result;

}

template <typename Matrix, typename MatrixTraits = matrix_traits<Matrix>>
bool lu_decomposition(Matrix& m) {

    typedef MatrixTraits mt;

    const size_t l = mt::rows(m);
    if (l != mt::cols(m)) {
        throw std::invalid_argument("lu_decomposition(..) available only for square matricies");
    }

    for (size_t i = 0; i < l - 1; i++) {
        std::pair<size_t, bool> pivot = find_lu_pivot(m, i);
        if (!pivot.second) {
            return false;
        }
        mt::swap_row(m, pivot.first, i);
        for (size_t k = i + 1; k < l; k++) {
            for (size_t j = i + 1; j < l; j++) {
                mt::cell(m, k, j) -= mt::cell(m, i, j) * mt::cell(m, k, i) / mt::cell(m, i, i);
            }
            mt::cell(m, k, i) /= mt::cell(m, i, i);
        }
    }
    return true;
}




}
