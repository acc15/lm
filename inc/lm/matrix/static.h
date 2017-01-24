#pragma once

#include <cstddef>

#include <lm/matrix/traits.h>

namespace lm {

template <size_t a, size_t b>
struct u_min {
    static const size_t value = a < b ? a : b;
};

template <size_t a, size_t b>
struct u_max {
    static const size_t value = a > b ? a : b;
};

template <typename M, typename N, size_t R, size_t C, typename Enable = void>
struct matrix_with_size {
    typedef N value_matrix_type;
};

template <typename M, typename N, size_t R, size_t C>
struct matrix_with_size<M, N, R, C, typename std::enable_if<R != 0 && C != 0>::type> {
    typedef typename M::template with_size<R, C>::value_matrix_type value_matrix_type;
};

template <typename M>
struct matrix_transpose {
    typedef typename matrix_with_size<M, M, M::Cols, M::Rows>::value_matrix_type value_matrix_type;
};

template <typename M, typename N, typename T = matrix_traits<N>>
struct matrix_product {
    typedef typename std::conditional<M::Rows != 0,
        typename matrix_with_size<M, N,
            u_min<M::Rows, T::Rows>::value,
            u_min<T::Cols, M::Cols>::value>::value_matrix_type,
        M>::type value_matrix_type;
};

}
