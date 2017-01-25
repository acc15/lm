#pragma once

#include <type_traits>

#include <lm/matrix/fwd.h>
#include <lm/matrix/type_util.h>

namespace lm {

template <template <class> class D, typename M>
struct matrix_decorator {

    typedef typename std::remove_reference<M>::type wrapped_type;
    typedef matrix<D<wrapped_type>> value_matrix_type;
    typedef matrix<D<wrapped_type&>> reference_matrix_type;
    typedef typename wrapped_type::value_type value_type;

    enum {
        Rows = wrapped_type::Rows,
        Cols = wrapped_type::Cols
    };

    template <size_t R, size_t C>
    struct with_size {
        typedef typename D<
            typename matrix_with_size<wrapped_type, wrapped_type, R, C>::value_matrix_type
        >::value_matrix_type value_matrix_type;
    };

};


}
