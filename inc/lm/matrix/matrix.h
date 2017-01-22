#pragma once

#include <cstddef>
#include <stdexcept>
#include <type_traits>

#include <lm/algorithm.h>
#include <lm/matrix/algorithm.h>
#include <lm/matrix/layout.h>
#include <lm/matrix/traits.h>


namespace lm {

template <typename S>
class matrix: public S {
public:

    typedef typename S::value_type value_type;
    typedef typename S::matrix_type matrix_type;
    typedef matrix<S> this_type;

    using S::S;

    value_type& cell(size_t row, size_t col) {
        return S::at(row, col);
    }

    const value_type& cell(size_t row, size_t col) const {
        return const_cast<this_type*>(this)->at(row, col);
    }

    value_type& operator()(size_t row, size_t col) {
        return cell(row, col);
    }

    const value_type& operator()(size_t row, size_t col) const {
        return cell(row, col);
    }

    template <typename T, typename Traits = matrix_traits<T>>
    this_type& assign(const T& other) {
        S::resize(Traits::rows(other), Traits::cols(other));
        apply<return_2nd, T, Traits>(other, return_2nd());
        return *this;
    }

    template <typename F, typename T, typename Traits = matrix_traits<T>>
    this_type& apply(const T& other, F func) {
        for (size_t i = 0; i < S::rows(); i++) {
            for (size_t j = 0; j < S::cols(); j++) {
                cell(i, j) = func(cell(i, j), Traits::cell(other, i, j));
            }
        }
        return *this;
    }

    template <typename T, typename Traits = matrix_traits<T>>
    this_type& add(const T& other) {
        return apply<std::plus<void>, T, Traits>(other, std::plus<void>());
    }

    template <typename T, typename Traits = matrix_traits<T>>
    this_type& subtract(const T& other) {
        return apply<std::minus<void>, T, Traits>(other, std::minus<void>());
    }

    template <typename T, typename Traits = matrix_traits<T>>
    bool equal(const T& other) {
        if (S::rows() != Traits::rows(other) || S::cols() != Traits::cols(other)) {
            return false;
        }
        for (size_t i = 0; i < S::rows(); i++) {
            for (size_t j = 0; j < S::cols(); j++) {
                if (cell(i, j) != Traits::cell(other, i, j)) {
                    return false;
                }
            }
        }
        return true;
    }

    template <typename T, typename Traits = matrix_traits<T>, typename P = typename matrix_product<matrix_type, T, Traits>::matrix_type>
    void product(const T& other, P& p) const {
        return lm::product<T, Traits, P>(*this, other, p);
    }

    template <typename T, typename Traits = matrix_traits<T>, typename P = typename matrix_product<matrix_type, T, Traits>::matrix_type>
    P product(const T& other) const {
        return lm::product<T, Traits, P>(*this, other);
    }

    template <typename P = typename matrix_transpose<matrix_type>::matrix_type>
    void compute_transposed(P& p) const {
        return lm::transpose<matrix_type, P>(*this, p);
    }

    template <typename P = typename matrix_transpose<matrix_type>::matrix_type>
    P compute_transposed() const {
        return lm::transpose<matrix_type, P>(*this);
    }

    template <typename P = typename matrix_transpose<matrix_type>::matrix_type,
               typename = typename std::enable_if<std::is_same<matrix_type, P>::value>::type>
    this_type& transpose() {
        P p;
        compute_transposed(p);
        return assign(p);
    }

    template <typename T, typename Traits = matrix_traits<T>>
    this_type& operator=(const T& other) {
        return assign<T, Traits>(other);
    }

    template <typename T, typename Traits = matrix_traits<T>>
    this_type& operator+=(const T& other) {
        return add<T, Traits>(other);
    }

    template <typename T, typename Traits = matrix_traits<T>>
    this_type& operator-=(const T& other) {
        return subtract<T, Traits>(other);
    }

    template <typename T, typename Traits = matrix_traits<T>>
    matrix_type operator+(const T& other) {
        return matrix_type(*this) += other;
    }

    template <typename T, typename Traits = matrix_traits<T>>
    matrix_type operator-(const T& other) {
        return matrix_type(*this) -= other;
    }

    template <typename T, typename Traits = matrix_traits<T>, typename P = typename matrix_product<matrix_type, T, Traits>::matrix_type>
    P operator*(const T& other) {
        return product<T, Traits, P>(other);
    }

    template <typename T, typename Traits = matrix_traits<T>>
    bool operator==(const T& other) {
        return equal<T, Traits>(other);
    }

    template <typename T, typename Traits = matrix_traits<T>>
    bool operator!=(const T& other) {
        return !equal<T, Traits>(other);
    }

};

template <typename M, typename MT = matrix_traits<M>>
class static_matrix_storage {
public:
    typedef typename MT::value_type value_type;

    enum {
        Rows = MT::Rows,
        Cols = MT::Cols
    };

    typedef typename std::remove_reference<M>::type storage_type;
    typedef matrix<static_matrix_storage<storage_type, MT>> matrix_type;
    typedef matrix<static_matrix_storage<storage_type&, MT>> ref;

    template <size_t Rows, size_t Cols>
    struct with_size {
        typedef typename MT::template with_size<Rows, Cols>::traits traits;
        typedef typename static_matrix_storage<typename traits::matrix_type, traits>::matrix_type matrix_type;
    };

    static_matrix_storage() = default;

    // initializer constructor
    template <typename T>
    static_matrix_storage(const std::initializer_list<std::initializer_list<T>>& m) {
        static_cast<matrix_type*>(this)->assign(m);
    }

    // initializer constructor
    template <typename T, typename Traits = range_matrix_traits<std::initializer_list<T>, Rows, Cols, row_major_layout>>
    static_matrix_storage(const std::initializer_list<T>& m) {
        static_cast<matrix_type*>(this)->template assign<const std::initializer_list<T>, Traits>(m);
    }

    // copy constructor
    template <typename T, typename Traits = matrix_traits<T>>
    static_matrix_storage(const T& other) {
        static_cast<matrix_type*>(this)->template assign<T, Traits>(other);
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
        Rows = 0,
        Cols = 0
    };

    typedef typename std::remove_reference<M>::type storage_type;
    typedef matrix<flat_dynamic_storage<storage_type, L>> matrix_type;
    typedef matrix<flat_dynamic_storage<storage_type&, L>> ref;

    flat_dynamic_storage() : _r(0), _c(0) {}

    // initializer constructor
    template <typename T>
    flat_dynamic_storage(const std::initializer_list<std::initializer_list<T>>& m) {
        static_cast<matrix_type*>(this)->assign(m);
    }

    // copy constructor
    template <typename T, typename Traits = matrix_traits<T>>
    flat_dynamic_storage(const T& other) {
        static_cast<matrix_type*>(this)->template assign<T, Traits>(other);
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


template <template <class> class D, typename M>
struct matrix_decorator {
    enum {
        Rows = M::Rows,
        Cols = M::Cols
    };

    typedef typename std::remove_reference<M>::type wrapped_type;
    typedef matrix<D<wrapped_type>> matrix_type;
    typedef matrix<D<wrapped_type&>> ref;
    typedef typename wrapped_type::value_type value_type;

    template <size_t R, size_t C>
    struct with_size {
        typedef typename D<
            typename matrix_with_size<wrapped_type, wrapped_type, R, C>::matrix_type
        >::matrix_type matrix_type;
    };

};

template <typename M>
class transposed_storage: public matrix_decorator<transposed_storage, M> {
public:
    typedef matrix_decorator<transposed_storage, M> base_type;
    typedef typename base_type::value_type value_type;

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
class permutation_storage: public matrix_decorator<permutation_storage, M> {
public:
    typedef matrix_decorator<permutation_storage, M> base_type;
    typedef typename base_type::value_type value_type;

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
        _c = 0;
        resize_p(rows());
        for (size_t i = 0; i < rows(); i++) {
            _p[i] = i;
        }
    }

    size_t permutation_count() const {
        return _c;
    }


private:
    struct static_vec {
        typedef size_t type[M::Rows];

        static void resize(type& t, size_t s) {
            if (s != M::Rows) {
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

    typedef typename std::conditional<M::Rows != 0, static_vec, dynamic_vec>::type vec_traits;

    void resize_p(size_t sz) {
        vec_traits::resize(_p, sz);
    }

    M& _m;
    typename vec_traits::type _p;
    size_t _c;

};

template <typename T, size_t Rows, size_t Cols>
using array_matrix = typename static_matrix_storage<T[Rows][Cols]>::matrix_type;

template <typename T, size_t Rows, size_t Cols, typename Layout = row_major_layout>
using flat_array_matrix = typename static_matrix_storage<T[Rows * Cols], array_matrix_traits<T, Rows, Cols, Layout>>::matrix_type;

template <template <class,size_t> class V, typename T, size_t Rows, size_t Cols, typename Layout = row_major_layout>
using container_matrix = typename static_matrix_storage<V<T,Rows*Cols>, container_matrix_traits<V,T,Rows,Cols,Layout>>::matrix_type;

template <typename M, typename MT = matrix_traits<M>>
using static_matrix = typename static_matrix_storage<M, MT>::matrix_type;

template <typename M, typename L = row_major_layout>
using flat_dynamic_matrix = typename flat_dynamic_storage<M, L>::matrix_type;

template <typename T, typename L = row_major_layout>
using vector_matrix = typename flat_dynamic_storage<std::vector<T>, L>::matrix_type;

template <typename M>
using transposed_matrix = typename transposed_storage<M>::matrix_type;

template <typename M>
using permutation_matrix = typename permutation_storage<M>::matrix_type;


}
