#pragma once

#include <cstddef>

namespace lm {

template <typename M>
struct matrix_traits {

};

template <typename T, size_t Rows, size_t Cols>
struct matrix_traits<T[Rows][Cols]> {

    static size_t rows(const T (&m) [Rows][Cols]) {
        return Rows;
    }

    static size_t cols(const T (&m) [Rows][Cols]) {
        return Cols;
    }

};

template <typename T>
struct vector_traits {

};

//template <typename T, size_t Size>
//struct vector_traits<T[Size]> {

//    static size_t size()

//}


template <typename Matrix, typename Vec>
bool lu_decomposition(Matrix& m, Vec& p) {

    typedef matrix_traits<Matrix> m_traits;
    typedef vector_traits<Vec> p_traits;


    const size_t cols = m_traits::cols(m);
    for (size_t i = 0; i < cols; i++) {
        p[i] = i;
    }

    return false;
}




}
