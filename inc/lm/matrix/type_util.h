#pragma once

#include <cstddef>

#include <lm/matrix/traits.h>

namespace lm {

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

template <typename M, typename N>
struct matrix_product {

    static_assert( M::Cols == 0 || N::Rows == 0 || M::Cols == static_cast<size_t>(N::Rows) , "matricies can't be multiplied" );

    typedef typename std::conditional<M::Rows != 0,
        typename matrix_with_size<M, N, M::Rows, N::Cols>::value_matrix_type,
        M>::type value_matrix_type;
};

}
