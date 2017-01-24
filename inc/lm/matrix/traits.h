#pragma once

#include <cstddef>
#include <lm/matrix/layout.h>

namespace lm {

template <typename T, typename Enable = void>
struct matrix_traits {

    typedef T container_type;
    typedef typename T::value_type value_type;

    enum {
        Rows = T::Rows,
        Cols = T::Cols
    };

    static size_t rows(const container_type& m) {
        return m.rows();
    }

    static size_t cols(const container_type& m) {
        return m.cols();
    }

    static const value_type& cell(const container_type& m, size_t row, size_t col) {
        return m.cell(row, col);
    }

    static value_type& cell(container_type& m, size_t row, size_t col) {
        return m.cell(row, col);
    }

};

template <typename T>
struct matrix_traits<T, typename std::enable_if<
        std::is_array<typename std::remove_reference<T>::type>::value &&
        std::rank<typename std::remove_reference<T>::type>::value == 2>::type> {

    typedef typename std::remove_reference<T>::type container_type;
    typedef typename std::remove_all_extents< container_type >::type value_type;

    enum {
        Rows = std::extent<container_type, 0>::value,
        Cols = std::extent<container_type, 1>::value
    };

    template <size_t Rows, size_t Cols>
    struct with_size {
        typedef matrix_traits<value_type[Rows][Cols]> traits;
    };

    static const value_type& cell(const container_type& m, size_t row, size_t col) {
        return m[row][col];
    }

    static value_type& cell(container_type& m, size_t row, size_t col) {
        return m[row][col];
    }

    static size_t rows(const container_type& m) {
        return Rows;
    }

    static size_t cols(const container_type& m) {
        return Cols;
    }

};

template <typename T>
struct matrix_traits<std::initializer_list<std::initializer_list<T>>> {

    typedef std::initializer_list<std::initializer_list<T>> container_type;
    typedef T value_type;

    static const value_type& cell(const container_type& m, size_t row, size_t col) {
        return *((m.begin() + row)->begin() + col);
    }

    static size_t rows(const container_type& m) {
        return m.size();
    }

    static size_t cols(const container_type& m) {
        return m.begin()->size();
    }


};

template <typename V, size_t R, size_t C, typename Layout = row_major_layout>
struct range_matrix_traits {

    typedef typename V::value_type value_type;
    typedef V container_type;

    enum {
        Rows = R,
        Cols = C
    };

    static const value_type& cell(container_type& m, size_t row, size_t col) {
        return *(m.begin() + Layout::compute_flat_index(row, col, R, C));
    }

    static const value_type& cell(const container_type& m, size_t row, size_t col) {
        return *(m.begin() + Layout::compute_flat_index(row, col, R, C));
    }

    static size_t rows(const container_type& m) {
        return Rows;
    }

    static size_t cols(const container_type& m) {
        return Cols;
    }

};

template <typename T, typename V, size_t R, size_t C, typename L>
struct flat_matrix_traits {

    typedef T container_type;
    typedef V value_type;

    enum {
        Rows = R,
        Cols = C
    };

    static const value_type& cell(const container_type& m, size_t row, size_t col) {
        return m[L::compute_flat_index(row, col, R, C)];
    }

    static value_type& cell(container_type& m, size_t row, size_t col) {
        return m[L::compute_flat_index(row, col, R, C)];
    }

    static size_t rows(const container_type& m) {
        return Rows;
    }

    static size_t cols(const container_type& m) {
        return Cols;
    }
};

template <template <class,size_t> class V, typename T, size_t R, size_t C, typename L = row_major_layout>
struct container_matrix_traits: public flat_matrix_traits<V<T,R*C>,T,R,C,L> {
    template <size_t Rows, size_t Cols>
    struct with_size {
        typedef container_matrix_traits<V, T, Rows, Cols, L> traits;
    };
};

template <typename T, size_t R, size_t C, typename L = row_major_layout>
struct array_matrix_traits: public flat_matrix_traits<T[R*C],T,R,C,L> {
    template <size_t Rows, size_t Cols>
    struct with_size {
        typedef array_matrix_traits<T, Rows, Cols, L> traits;
    };
};


}
