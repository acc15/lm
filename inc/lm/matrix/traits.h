#pragma once

#include <cstddef>
#include <lm/matrix/layout.h>

namespace lm {

template <typename T, typename Enable = void>
struct matrix_traits {

    typedef typename T::value_type value_type;

    enum {
        Static = T::Static,
        Rows = T::Rows,
        Cols = T::Cols
    };

    static size_t rows(const T& m) {
        return m.rows();
    }

    static size_t cols(const T& m) {
        return m.cols();
    }

    static const value_type& cell(const T& m, size_t row, size_t col) {
        return m.cell(row, col);
    }

    static value_type& cell(T& m, size_t row, size_t col) {
        return m.cell(row, col);
    }

};

template <typename T>
struct matrix_traits<T, typename std::enable_if<
        std::is_array<typename std::remove_reference<T>::type>::value &&
        std::rank<typename std::remove_reference<T>::type>::value == 2>::type> {

    typedef typename std::remove_reference<T>::type matrix_type;
    typedef typename std::remove_all_extents< matrix_type >::type value_type;

    enum {
        Static = true,
        Rows = std::extent<matrix_type, 0>::value,
        Cols = std::extent<matrix_type, 1>::value
    };

    template <size_t Rows, size_t Cols>
    struct with_size {
        typedef matrix_traits<value_type[Rows][Cols]> traits;
    };

    static const value_type& cell(const matrix_type& m, size_t row, size_t col) {
        return m[row][col];
    }

    static value_type& cell(matrix_type& m, size_t row, size_t col) {
        return m[row][col];
    }

    static size_t rows(const matrix_type& m) {
        return Rows;
    }

    static size_t cols(const matrix_type& m) {
        return Cols;
    }

};

template <typename T>
struct matrix_traits<std::initializer_list<std::initializer_list<T>>> {

    typedef std::initializer_list<std::initializer_list<T>> matrix_type;
    typedef T value_type;

    static const value_type& cell(const matrix_type& m, size_t row, size_t col) {
        return *((m.begin() + row)->begin() + col);
    }

    static size_t rows(const matrix_type& m) {
        return m.size();
    }

    static size_t cols(const matrix_type& m) {
        return m.begin()->size();
    }


};

template <typename T, size_t R, size_t C, typename Layout = row_major_layout>
struct initializer_matrix_traits {

    typedef T value_type;
    typedef std::initializer_list<T> matrix_type;

    enum {
        Static = true,
        Rows = R,
        Cols = C
    };

//    template <size_t Rows, size_t Cols>
//    struct with_size {
//        typedef flat_matrix_traits<T, Rows, Cols, Layout> traits;
//    };

    static const value_type& cell(const matrix_type& m, size_t row, size_t col) {
        return *(m.begin() + Layout::compute_flat_index(row, col, R, C));
    }

    static size_t rows(const matrix_type& m) {
        return Rows;
    }

    static size_t cols(const matrix_type& m) {
        return Cols;
    }

};


template <typename T, size_t R, size_t C, typename Layout = row_major_layout>
struct array_matrix_traits {

    typedef T value_type;
    typedef T matrix_type[R*C];

    enum {
        Static = true,
        Rows = R,
        Cols = C
    };

    template <size_t Rows, size_t Cols>
    struct with_size {
        typedef array_matrix_traits<T, Rows, Cols, Layout> traits;
    };

    static const value_type& cell(const matrix_type& m, size_t row, size_t col) {
        return m[Layout::compute_flat_index(row, col, R, C)];
    }

    static value_type& cell(matrix_type& m, size_t row, size_t col) {
        return m[Layout::compute_flat_index(row, col, R, C)];
    }

    static size_t rows(const matrix_type& m) {
        return Rows;
    }

    static size_t cols(const matrix_type& m) {
        return Cols;
    }

};


}
