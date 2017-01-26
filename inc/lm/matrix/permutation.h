#pragma once

#include <cstddef>
#include <cassert>

#include <numeric>

#include <lm/matrix/decorator.h>

namespace lm {

template <size_t Size>
struct permutation_vec_traits {
    typedef size_t type[Size];

    static void resize(type& t, size_t s) {
        assert( s == Size );
    }
};

template<>
struct permutation_vec_traits<0> {
    typedef std::vector<size_t> type;

    static void resize(type& t, size_t s) {
        t.resize(s);
    }
};


template <typename M>
class permutation_storage: public matrix_decorator<permutation_storage, M> {
public:
    typedef matrix_decorator<::lm::permutation_storage, M> base_type;
    typedef typename base_type::value_type value_type;

    typedef permutation_vec_traits<base_type::Rows> vec_traits;
    typedef typename vec_traits::type pm;

    permutation_storage()  {
        reset();
    }

    template <typename T>
    permutation_storage(const std::initializer_list<std::initializer_list<T>>& other) : _m(other) {
        reset();
    }

    template <typename T> permutation_storage(const std::initializer_list<T>& other) : _m(other) {
        reset();
    }

    template <typename T> permutation_storage(const T& other) : _m(other) {
        reset();
    }

    template <typename T = M, typename = typename std::enable_if<std::is_reference<T>::value && std::is_same<T, M>::value>::type>
    permutation_storage(M ref) : _m(ref) {
        reset();
    }

    size_t rows() const { return _m.rows(); }
    size_t cols() const { return _m.cols(); }

    value_type& at(size_t row, size_t col) {
        return _m.at(_p[row], col);
    }

    void resize(size_t r, size_t c) {
        if (r == rows() && c == cols()) {
            return;
        }
        _m.resize(r, c);
        reset();
    }

    void swap_row(size_t r1, size_t r2) {
        if (r1 != r2) {
            std::swap(_p[r1], _p[r2]);
            ++_c;
        }
    }

    void swap_col(size_t c1, size_t c2) {
        for (size_t i = 0; i < rows(); i++) {
            std::swap(_m.at(_p[i], c1), _m.at(_p[i], c1));
        }
    }

    void reset() {
        _c = 0;
        resize_p(rows());
        std::iota(std::begin(_p), std::end(_p), 0);
    }

    size_t permutation_count() const {
        return _c;
    }

    const pm& permutation_vec() const {
        return _p;
    }

private:
    void resize_p(size_t sz) {
        vec_traits::resize(_p, sz);
    }

    M _m;
    pm _p;
    size_t _c;

};

template <typename M> using permutation_matrix = typename permutation_storage<M>::value_matrix_type;


}
