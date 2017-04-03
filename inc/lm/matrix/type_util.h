#pragma once

#include <cstddef>

#include <lm/matrix/traits.h>

namespace lm {

template <typename M, size_t R, size_t C, typename Enable = void>
struct matrix_with_size {
    typedef M value_matrix_type;
};

template <typename M, size_t R, size_t C>
struct matrix_with_size<M, R, C, typename std::enable_if<R != 0 && C != 0>::type> {
    typedef typename M::template with_size<R, C>::value_matrix_type value_matrix_type;
};

template <typename M>
struct matrix_transpose {
    typedef typename matrix_with_size<M, M::Cols, M::Rows>::value_matrix_type value_matrix_type;
};

template <typename M, typename N>
struct matrix_product {
    typedef typename std::conditional<M::Rows != 0, // M is static?
        typename std::conditional<N::Cols != 0, // N is static?
            typename matrix_with_size<M,
                (M::Rows < N::Rows ? M::Rows : N::Rows),
                (N::Cols < M::Cols ? N::Cols : M::Cols)>::value_matrix_type, // both is static
            N // N is dynamic
        >::type,
        M // M is dynamic
    >::type value_matrix_type;
};

}
