#pragma once

#include <cstddef>
#include <type_traits>

#include <lm/matrix/fwd.h>
#include <lm/matrix/layout.h>

namespace lm {

template <typename M, typename L = row_major_layout>
class flat_dynamic_storage {
public:
    typedef typename M::value_type value_type;

    enum {
        Rows = 0,
        Cols = 0
    };

    typedef typename std::remove_reference<M>::type storage_type;
    typedef matrix<flat_dynamic_storage<storage_type, L>> value_matrix_type;
    typedef matrix<flat_dynamic_storage<storage_type&, L>> reference_matrix_type;

    flat_dynamic_storage() : _r(0), _c(0) {}

    // initializer constructor
    template <typename T>
    flat_dynamic_storage(const std::initializer_list<std::initializer_list<T>>& m) {
        static_cast<value_matrix_type*>(this)->assign(m);
    }

    // copy constructor
    template <typename T>
    flat_dynamic_storage(const T& other) {
        static_cast<value_matrix_type*>(this)->assign(other);
    }

    flat_dynamic_storage(size_t r, size_t c) {
        resize(r, c);
    }

    // reference constructor
    template <typename T = M, typename = typename std::enable_if<std::is_reference<M>::value && std::is_same<T, M>::value>::type>
    flat_dynamic_storage(M m, size_t r = 0, size_t c = 0) : _m(m) {
        resize(r, c);
    }

    size_t rows() const {
        return _r;
    }

    size_t cols() const {
        return _c;
    }

    value_type& at(size_t row, size_t col) {
        return _m[L::compute_flat_index(row, col, _r, _c)];
    }

    void swap_row(size_t r1, size_t r2) {
        lm::swap_row(*this, r1, r2);
    }

    void swap_col(size_t c1, size_t c2) {
        lm::swap_row(*this, c1, c2);
    }

    void resize(size_t rows, size_t cols) {
        _m.resize(rows * cols);
        _r = rows;
        _c = cols;
    }

private:
    M _m;
    size_t _r, _c;

};

template <typename M, typename L = row_major_layout>
using flat_dynamic_matrix = typename flat_dynamic_storage<M, L>::value_matrix_type;

template <typename T, typename L = row_major_layout>
using vector_matrix = flat_dynamic_matrix<std::vector<T>, L>;



}
