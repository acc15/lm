#pragma once

#include <cstddef>
#include <type_traits>

#include <lm/util/assert.h>
#include <lm/matrix/fwd.h>
#include <lm/matrix/traits.h>
#include <lm/matrix/layout.h>
#include <lm/matrix/algorithm.h>

namespace lm {


template <typename M, typename MT = matrix_traits<M>>
class static_matrix_storage {
public:
    typedef typename MT::value_type value_type;

    constexpr static size_t Rows = MT::Rows;
    constexpr static size_t Cols = MT::Cols;

    typedef typename std::remove_reference<M>::type storage_type;

    typedef matrix<static_matrix_storage<storage_type, MT>> value_matrix_type;
    typedef matrix<static_matrix_storage<storage_type&, MT>> reference_matrix_type;

    template <size_t Rows, size_t Cols>
    struct with_size {
        typedef typename MT::template with_size<Rows, Cols>::traits traits;
        typedef typename static_matrix_storage<typename traits::container_type, traits>::value_matrix_type value_matrix_type;
    };

    static_matrix_storage() = default;

    // initializer constructor
    template <typename T>
    static_matrix_storage(const std::initializer_list<std::initializer_list<T>>& m) {
        static_cast<value_matrix_type*>(this)->assign(m);
    }

    // initializer constructor
    template <typename T>
    static_matrix_storage(const std::initializer_list<T>& m) {
        static_cast<value_matrix_type*>(this)->template assign<const std::initializer_list<T>,
                range_matrix_traits<std::initializer_list<T>, Rows, Cols, row_major_layout>>(m);
    }

    // copy constructor
    template <typename T>
    static_matrix_storage(const T& other) {
        static_cast<value_matrix_type*>(this)->assign(other);
    }

    // reference constructor
    template <typename T = M, typename = typename std::enable_if<std::is_reference<M>::value && std::is_same<T, M>::value>::type>
    static_matrix_storage(M m) : _m(m) {
    }

    size_t rows() const {
        return MT::rows(_m);
    }

    size_t cols() const {
        return MT::cols(_m);
    }

    void swap_row(size_t r1, size_t r2) {
        lm::swap_row(*this, r1, r2);
    }

    void swap_col(size_t c1, size_t c2) {
        lm::swap_row(*this, c1, c2);
    }

    value_type& at(size_t row, size_t col) {
        return MT::cell(_m, row, col);
    }

    void resize(size_t rows, size_t cols) {
        lm_assert( rows == Rows && cols == Cols, "static matricies can't be resized" );
    }

    const storage_type& value() const {
        return _m;
    }

    storage_type& value() {
        return _m;
    }

private:
    M _m;

};

template <typename M, typename MT = matrix_traits<M>>
using static_matrix = typename static_matrix_storage<M, MT>::value_matrix_type;

template <typename T, size_t Rows, size_t Cols>
using array_matrix = static_matrix<T[Rows][Cols]>;

template <typename T, size_t Rows, size_t Cols, typename Layout = row_major_layout>
using flat_array_matrix = static_matrix<T[Rows * Cols], array_matrix_traits<T, Rows, Cols, Layout>>;

template <template <class, size_t> class V, typename T, size_t Rows, size_t Cols, typename Layout = row_major_layout>
using container_matrix = static_matrix<V<T,Rows*Cols>, container_matrix_traits<V,T,Rows,Cols,Layout>>;


}
