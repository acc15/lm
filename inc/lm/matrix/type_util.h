#pragma once

#include <cstddef>
#include <type_traits>

#include <lm/matrix/traits.h>

namespace lm {

template <typename M, size_t R, size_t C, typename Enable = void>
struct matrix_with_size {
    typedef M value_matrix_type;
};

template <typename M, size_t R, size_t C>
struct matrix_with_size<M, R, C, typename std::enable_if<!M::resizable>::type> {
    typedef typename M::template with_size<R, C>::value_matrix_type value_matrix_type;
};

template <typename M>
struct matrix_transpose {

    typedef typename std::conditional<M::resizable,
        M,
        typename matrix_with_size<M, M().cols(), M().rows()>::value_matrix_type
    >::type value_matrix_type;

    //typedef typename matrix_with_size<M, M::Cols, M::Rows>::value_matrix_type value_matrix_type;
};

template <typename M, typename N>
struct matrix_product {
    typedef typename std::conditional<M::resizable,
        M,
        typename std::conditional<N::resizable,
            N,
            typename matrix_with_size<M, M().rows(), N().cols()>::value_matrix_type
        >::type
    >::type value_matrix_type;
};

}
