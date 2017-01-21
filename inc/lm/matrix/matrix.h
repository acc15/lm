#pragma once

#include <cstddef>
#include <stdexcept>
#include <type_traits>

#include <lm/algorithm.h>
#include <lm/matrix/algorithms.h>
#include <lm/matrix/layout.h>
#include <lm/matrix/traits.h>


namespace lm {

template <typename S>
class matrix: public S {
public:

    typedef typename S::value_type value_type;
    typedef matrix<S> matrix_type;

    using S::S;

    value_type& cell(size_t row, size_t col) {
        return S::at(row, col);
    }

    const value_type& cell(size_t row, size_t col) const {
        return const_cast<matrix_type*>(this)->at(row, col);
    }

    value_type& operator()(size_t row, size_t col) {
        return cell(row, col);
    }

    const value_type& operator()(size_t row, size_t col) const {
        return cell(row, col);
    }

    template <typename T, typename Traits = matrix_traits<T>>
    matrix_type& assign(const T& other) {
        S::resize(Traits::rows(other), Traits::cols(other));
        apply<return_2nd, T, Traits>(other, return_2nd());
        return *this;
    }

    template <typename F, typename T, typename Traits = matrix_traits<T>>
    matrix_type& apply(const T& other, F func) {
        for (size_t i = 0; i < S::rows(); i++) {
            for (size_t j = 0; j < S::cols(); j++) {
                cell(i, j) = func(cell(i, j), Traits::cell(other, i, j));
            }
        }
        return *this;
    }

    template <typename T, typename Traits = matrix_traits<T>>
    matrix_type& add(const T& other) {
        return apply<std::plus, T, Traits>(other, std::plus<void>());
    }

    template <typename T, typename Traits = matrix_traits<T>>
    matrix_type& subtract(const T& other) {
        return apply<std::minus, T, Traits>(other, std::minus<void>());
    }

//    template <typename T, typename Traits = matrix_traits<T>>
//    typename std::conditional<matrix_traits::Static,
//        std::conditional<Traits::Static,
//            matrix_traits::with_size<matrix_traits::Rows, Traits::Cols>::type,
//            Traits::type>::type,
//        matrix_traits::type>::type product(const T& other) const {

//        typename std::conditional<matrix_traits::Static,
//            std::conditional<Traits::Static,
//                matrix_traits::with_size<matrix_traits::Rows, Traits::Cols>::type,
//                Traits::type>::type,
//            matrix_traits::type>::type result;

//        result.resize(rows(), other.cols());
//        for (size_t i = 0; i < result.rows(); i++) {
//            for (size_t j = 0; j < result.cols(); j++) {
//                value_type sum = 0;
//                for (size_t k = 0; k < cols(); k++) {
//                    sum += cell(i, k) * other(k, j);
//                }
//                result(i, j) = sum;
//            }
//        }
//        return result;
//    }

    template <typename T, typename Traits = matrix_traits<T>>
    matrix_type& operator=(const T& other) {
        return assign<T, Traits>(other);
    }

    template <typename T, typename Traits = matrix_traits<T>>
    matrix_type& operator+=(const T& other) {
        return add<T, Traits>(other);
    }

    template <typename T, typename Traits = matrix_traits<T>>
    matrix_type& operator-=(const T& other) {
        return subtract<T, Traits>(other);
    }

};

template <typename M, typename MT = matrix_traits<M>>
class static_matrix_storage {
public:
    typedef typename MT::value_type value_type;

    enum {
        Static = MT::Static,
        Rows = MT::Rows,
        Cols = MT::Cols
    };

    static_matrix_storage() = default;

    // copy constructor
    template <typename T, typename Traits = matrix_traits<T>>
    static_matrix_storage(const T& other) {
        static_cast<matrix<static_matrix_storage<M, MT>>*>(this)->template assign<T, Traits>(other);
    }

    // reference constructor
    template <typename T = M, typename = typename std::enable_if<std::is_reference<M>::value && std::is_same<T, M>::value>::type>
    static_matrix_storage(M m) : _m(m) {
    }

    template <size_t Rows, size_t Cols>
    struct with_size {
        typedef static_matrix_storage<
            typename MT::template with_size<Rows, Cols>::traits::matrix_type,
            typename MT::template with_size<Rows, Cols>::traits> matrix_type;
    };

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
        if (rows != Rows || cols != Cols) {
            throw std::logic_error("static_matrix can't be resized");
        }
    }

private:
    M _m;

};


template <typename M, typename L = row_major_layout>
class flat_dynamic_storage {
public:
    typedef typename M::value_type value_type;

    enum {
        Static = false,
        Rows = 0,
        Cols = 0
    };

    flat_dynamic_storage() : _r(0), _c(0) {}

    // copy constructor
    template <typename T, typename Traits>
    flat_dynamic_storage(const T& other) {
        static_cast<matrix<flat_dynamic_storage<M, L>>*>(this)->template assign<T, Traits>(other);
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

template <typename M>
class transposed_storage {
public:
    typedef typename M::value_type value_type;

    enum {
        Static = M::Static,
        Rows = M::Rows,
        Cols = M::Cols
    };

    transposed_storage(M& ref) : _m(ref) {
    }

    size_t rows() { return _m.cols(); }
    size_t cols() { return _m.rows(); }

    value_type& at(size_t row, size_t col) {
        return _m.at(col, row);
    }

    void resize(size_t rows, size_t cols) {
        _m.resize(cols, rows);
    }

    void swap_row(size_t r1, size_t r2) {
        _m.swap_col(r1, r2);
    }

    void swap_col(size_t c1, size_t c2) {
        _m.swap_row(c1, c2);
    }

private:
    M& _m;

};

template <typename M>
class permutation_storage {
public:
    typedef typename M::value_type value_type;

    enum {
        Static = M::Static,
        Rows = M::Rows,
        Cols = M::Cols
    };

    permutation_storage(M& ref) : _m(ref) {
        reset();
    }

    size_t rows() { return _m.rows(); }
    size_t cols() { return _m.cols(); }

    value_type& at(size_t row, size_t col) {
        return _m.at(_p[row], col);
    }

    void resize(size_t rows, size_t cols) {
        _m.resize(rows, cols);
        resize_p(rows);
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
        resize_p(rows());
        for (size_t i = 0; i < rows(); i++) {
            _p[i] = i;
        }
        _c = 0;

    }

    size_t permutation_count() const {
        return _c;
    }


private:
    struct static_vec {
        typedef size_t type[Rows];

        static void resize(type& t, size_t s) {
            if (s != Rows) {
                throw std::length_error("static vec can't be resized");
            }
        }
    };

    struct dynamic_vec {
        typedef std::vector<size_t> type;

        static void resize(type& t, size_t s) {
            t.resize(s);
        }
    };

    typedef typename std::conditional<Static, static_vec, dynamic_vec>::type vec_traits;

    void resize_p(size_t sz) {
        vec_traits::resize(_p, sz);
    }

    M& _m;
    typename vec_traits::type _p;
    size_t _c;

};

template <typename T, size_t Rows, size_t Cols>
using array_matrix = matrix< static_matrix_storage<T[Rows][Cols]> >;

template <typename T, size_t Rows, size_t Cols, typename Layout = row_major_layout>
using flat_array_matrix = matrix< static_matrix_storage<T[Rows * Cols], array_matrix_traits<T, Rows, Cols, Layout>> >;

template <typename M, typename MT = matrix_traits<M>>
using static_matrix = matrix< static_matrix_storage<M, MT> >;

template <typename M, typename L = row_major_layout>
using flat_dynamic_matrix = matrix< flat_dynamic_storage<M, L> >;

template <typename M>
using transposed_matrix = matrix< transposed_storage<M> >;

template <typename M>
using permutation_matrix = matrix< permutation_storage<M> >;


}
