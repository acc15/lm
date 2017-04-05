#pragma once

#include <type_traits>

#include <lm/matrix/fwd.h>
#include <lm/matrix/type_util.h>

namespace lm {

/**
 * @brief base class for matrix decorators
 *
 */
template <template <class> class D, typename M>
struct matrix_decorator {

    typedef typename std::remove_reference<M>::type storage_type;
    typedef matrix<D<storage_type>> value_matrix_type;
    typedef matrix<D<storage_type&>> reference_matrix_type;
    typedef typename storage_type::value_type value_type;

//    constexpr static size_t Rows = storage_type::Rows;
//    constexpr static size_t Cols = storage_type::Cols;

    template <size_t R, size_t C>
    struct with_size {
        typedef typename D<
            typename matrix_with_size<storage_type, R, C>::value_matrix_type
        >::value_matrix_type value_matrix_type;
    };

//    constexpr size_t rows() const {
//        return
//    }

};


}
