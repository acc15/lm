#pragma once

#include <cstddef>
#include <lm/matrix/traits.h>

namespace lm {

template <typename M>
void swap_row(M& m, size_t r1, size_t r2) {
    if (r1 == r2) {
        return;
    }
    for (size_t i = 0; i < m.cols(); i++) {
        std::swap(m.at(r1, i), m.at(r2, i));
    }
}

template <typename M>
void swap_col(M& m, size_t c1, size_t c2) {
    if (c1 == c2) {
        return;
    }
    for (size_t i = 0; i < m.rows(); i++) {
        std::swap(m.at(i, c1), m.at(i, c2));
    }
}

template <typename M, typename N, typename T = matrix_traits<N>>
struct matrix_product {

    template <size_t R, size_t C, bool Enable = false>
    struct static_product {
        typedef N matrix_type;
    };

    template <size_t R, size_t C>
    struct static_product<R, C, true> {
        typedef typename M::template with_size<R, C>::matrix_type matrix_type;
    };

    typedef typename std::conditional<M::Rows != 0,
        typename static_product<M::Rows, T::Cols, M::Rows != 0 && T::Rows != 0>::matrix_type,
        M>::type matrix_type;

};


template <typename M, typename N, typename T = matrix_traits<N>, typename P = typename matrix_product<M, N, T>::matrix_type>
P product(const M& m, const N& n) {

    P result;
    result.resize(m.rows(), T::cols(n));
    for (size_t i = 0; i < result.rows(); i++) {
        for (size_t j = 0; j < result.cols(); j++) {
            typename M::value_type sum = 0;
            for (size_t k = 0; k < m.cols(); k++) {
                sum += m(i, k) * T::cell(n, k, j);
            }
            result(i, j) = sum;
        }
    }
    return result;
}

template <typename M>
std::pair<size_t, bool> find_lu_pivot(M& m, const size_t n) {

    size_t l = m.rows();

    std::pair<size_t, bool> result(0, false);
    for (size_t i = n; i < l; i++) {

        const typename M::value_type v = m(i, n);
        if (v == 0) {
            continue;
        }

        if (!result.second) {
            result.first = i;
            result.second = true;
            continue;
        }

        if (std::abs(v) < std::abs(m(result.first, n))) {
            result.first = i;
        }

    }
    return result;

}

template <typename M>
bool lu_decomposition(M& m) {

    const size_t l = m.rows();
    if (l != m.cols()) {
        throw std::invalid_argument("lu_decomposition(..) available only for square matricies");
    }

    for (size_t i = 0; i < l - 1; i++) {
        std::pair<size_t, bool> pivot = find_lu_pivot(m, i);
        if (!pivot.second) {
            return false;
        }
        m.swap_row(pivot.first, i);
        for (size_t k = i + 1; k < l; k++) {
            for (size_t j = i + 1; j < l; j++) {
                m(k, j) -= m(i, j) * m(k, i) / m(i, i);
            }
            m(k, i) /= m(i, i);
        }
    }
    return true;
}

}
